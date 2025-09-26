# simulation-farm/runners/k8s_runner.py
"""
K8sRunner: submit Simulation Farm jobs to a Kubernetes cluster.

- Generates Kubernetes Job manifests dynamically from spec.
- Uses official kubernetes Python client (pip install kubernetes).
- Supports resource requests/limits, namespace, image, env/volumes.
- Collects logs from pods and returns job status.

Typical usage:
--------------
from simulation_farm.runners.k8s_runner import K8sRunner
from simulation_farm.jobs.backtest_job import BacktestSpec, BacktestJob

spec = BacktestSpec(
    run_id="bt_k8s_demo",
    data_path="s3://mybucket/sp500_daily.parquet",
    strategy="momentum",
    params={"lookback_days": 90},
    start="2020-01-01", end="2023-12-31",
)

runner = K8sRunner(namespace="simfarm", image="simfarm:latest")
job_name = runner.submit("backtest", spec)
status = runner.wait(job_name)
print(status)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from kubernetes import client, config, utils
from kubernetes.stream import stream


class K8sRunner:
    def __init__(
        self,
        namespace: str = "default",
        image: str = "simfarm:latest",
        backoff_limit: int = 2,
        ttl_seconds_after_finished: int = 600,
        resources: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        """
        Args:
            namespace: Kubernetes namespace.
            image: container image to run (must include simfarm code + deps).
            backoff_limit: Job retry attempts.
            ttl_seconds_after_finished: when to GC completed jobs.
            resources: dict of {limits:{cpu,memory}, requests:{cpu,memory}}.
        """
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        self.api = client.BatchV1Api()
        self.core = client.CoreV1Api()
        self.namespace = namespace
        self.image = image
        self.backoff_limit = backoff_limit
        self.ttl = ttl_seconds_after_finished
        self.resources = resources or {
            "limits": {"cpu": "2", "memory": "4Gi"},
            "requests": {"cpu": "1", "memory": "2Gi"},
        }

    def _make_job(self, name: str, command: list[str], env: Optional[Dict[str, str]] = None) -> client.V1Job:
        """Build a Job manifest."""
        env_list = [client.V1EnvVar(name=k, value=v) for k, v in (env or {}).items()]

        container = client.V1Container(
            name="runner",
            image=self.image,
            command=command,
            env=env_list,
            resources=client.V1ResourceRequirements(**self.resources),
        )
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"job-name": name}),
            spec=client.V1PodSpec(restart_policy="Never", containers=[container]),
        )
        job_spec = client.V1JobSpec(
            ttl_seconds_after_finished=self.ttl,
            backoff_limit=self.backoff_limit,
            template=template,
        )
        return client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=name),
            spec=job_spec,
        )

    def submit(self, job_type: str, spec: Any, env: Optional[Dict[str, str]] = None) -> str:
        """
        Submit a Simulation Farm job to Kubernetes.

        Args:
            job_type: "backtest" | "monte_carlo" | "replay" | "stress_test"
            spec: dataclass spec for the job (BacktestSpec, MonteCarloSpec, etc.)
            env: extra env vars (dict)

        Returns:
            job_name (str)
        """
        job_name = f"simfarm-{job_type}-{spec.run_id}".lower().replace("_", "-")
        # command: run via module CLI
        module_map = {
            "backtest": "simulation_farm.jobs.backtest_job",
            "monte_carlo": "simulation_farm.jobs.monte_carlo_job",
            "replay": "simulation_farm.jobs.replay_job",
            "stress_test": "simulation_farm.jobs.stress_test_job",
        }
        if job_type not in module_map:
            raise ValueError(f"Unknown job_type: {job_type}")
        cmd = [
            "python",
            "-m",
            module_map[job_type],
            "--run-id", spec.run_id,
            "--data", spec.data_path,
        ]
        # add params if present
        if getattr(spec, "start", None): cmd += ["--start", spec.start]
        if getattr(spec, "end", None): cmd += ["--end", spec.end]
        if getattr(spec, "cash", None): cmd += ["--cash", str(spec.cash)]

        job = self._make_job(job_name, cmd, env)
        self.api.create_namespaced_job(namespace=self.namespace, body=job)
        return job_name

    def wait(self, job_name: str, timeout: int = 3600, poll: int = 10) -> Dict[str, Any]:
        """
        Wait for job to complete, stream logs, and return final status.
        """
        start = time.time()
        while time.time() - start < timeout:
            job = self.api.read_namespaced_job(job_name, self.namespace)
            if job.status.succeeded: # type: ignore
                pods = self.core.list_namespaced_pod(self.namespace, label_selector=f"job-name={job_name}")
                logs = []
                for pod in pods.items:
                    try:
                        log = self.core.read_namespaced_pod_log(pod.metadata.name, self.namespace)
                        logs.append(log)
                    except Exception:
                        pass
                return {"status": "succeeded", "logs": logs}
            if job.status.failed: # type: ignore
                return {"status": "failed"}
            time.sleep(poll)
        return {"status": "timeout"}