"use client";

import React from "react";

type Props = {
  children: React.ReactNode;
  /** Optional custom fallback UI */
  fallback?: React.ReactNode;
};

type State = { hasError: boolean; error: Error | null };

export default class ErrorBoundary extends React.Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // You can log to a service here if needed
    console.error("ErrorBoundary caught:", error, info);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;

      // Default fallback
      return (
        <div className="w-full rounded-lg border border-red-300 bg-red-50 p-4 text-red-800">
          <h2 className="font-semibold">Something went wrong.</h2>
          <p className="mt-1 text-sm">
            {this.state.error?.message ?? "Unknown error"}
          </p>
          <button
            onClick={this.handleReset}
            className="mt-3 rounded bg-red-600 px-3 py-1.5 text-sm text-white hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}