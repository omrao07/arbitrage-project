import { useState, useEffect } from 'react'
import axios from 'axios'

const API_BASE = 'http://localhost:8000' // change if hosted elsewhere

export const useHoldings = () => {
  const [holdings, setHoldings] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchHoldings = async () => {
    try {
      const response = await axios.get(`${API_BASE}/pnl/holdings`)
      setHoldings(response.data)
    } catch (err) {
      setError(err.message || 'Error fetching holdings')
    } finally {
      setLoading(false)
    }
  }

  const updateHolding = async (strategyId, position) => {
    try {
      await axios.post(`${API_BASE}/pnl/update`, {
        strategy: strategyId,
        position,
      })
      await fetchHoldings() // refresh state after update
    } catch (err) {
      setError(err.message || 'Error updating holding')
    }
  }

  const resetHoldings = async () => {
    try {
      await axios.post(`${API_BASE}/pnl/reset`)
      await fetchHoldings()
    } catch (err) {
      setError(err.message || 'Error resetting holdings')
    }
  }

  useEffect(() => {
    fetchHoldings()
  }, [])

  return {
    holdings,
    loading,
    error,
    updateHolding,
    resetHoldings,
    refreshHoldings: fetchHoldings,
  }
}