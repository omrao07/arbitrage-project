import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000'; // Replace with deployed backend URL if needed

export const useToggle = () => {
  const [toggles, setToggles] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchToggles = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${API_BASE}/strategy-toggles`);
      setToggles(res.data); // { strategy_name: true/false }
    } catch (err) {
      setError(err.message || 'Failed to fetch toggles');
    } finally {
      setLoading(false);
    }
  };

  const updateToggle = async (strategy, enabled) => {
    try {
      await axios.post(`${API_BASE}/strategy-toggles`, {
        strategy,
        enabled,
      });
      setToggles((prev) => ({
        ...prev,
        [strategy]: enabled,
      }));
    } catch (err) {
      setError(err.message || 'Failed to update toggle');
    }
  };

  useEffect(() => {
    fetchToggles();
  }, []);

  return {
    toggles,
    loading,
    error,
    updateToggle,
    refreshToggles: fetchToggles,
  };
};