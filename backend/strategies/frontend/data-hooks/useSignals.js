import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000'; // Change to your backend URL if deployed

export const useSignals = () => {
  const [signals, setSignals] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchSignals = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE}/signals`);
      setSignals(response.data); // expected to be a dictionary of signals per strategy
    } catch (err) {
      setError(err.message || 'Error fetching signals');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSignals();
  }, []);

  return {
    signals,
    loading,
    error,
    refreshSignals: fetchSignals,
  };
};