import { useState, useEffect } from 'react'
import axios from 'axios'

const API_BASE = 'http://localhost:8000' // Update this if hosted remotely

export const usePnL = () => {
  const [pnlData, setPnlData] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchPnL = async () => {
    setLoading(true)
    try {
      const response = await axios.get(`${API_BASE}/pnl`)
      setPnlData(response.data)
    } catch (err) {
      setError(err.message || 'Error fetching PnL data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPnL()
  }, [])

  return {
    pnlData,
    loading,
    error,
    refreshPnL: fetchPnL,
  }
}