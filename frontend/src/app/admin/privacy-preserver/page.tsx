'use client'

import { useState, useEffect } from 'react'
import { Eye, Lock, Shield, AlertTriangle, ArrowLeft } from 'lucide-react'
import Link from 'next/link'
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface PIIDetectionLog {
  id: string
  timestamp: string
  original_content: string
  masked_content: string
  pii_mappings: Record<string, { original: string; type: string; strategy: string; confidence: number }>
  conversation_id: string
  explanation: string | null
}

interface PrivacyPreserverStats {
  total_pii_detections: number
  pii_by_type: Record<string, number>
  masked_messages: number
  recent_pii_detections: number
  pii_detection_rate: number
  total_messages: number
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d']

export default function PrivacyPreserverDashboard() {
  const [stats, setStats] = useState<PrivacyPreserverStats | null>(null)
  const [logs, setLogs] = useState<PIIDetectionLog[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Add timestamp to force fresh request
        const timestamp = Date.now()
        // Fetch stats
        const statsRes = await fetch(`/api/admin/privacy-preserver/stats?_t=${timestamp}`, {
          cache: 'no-store',
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          }
        })
        if (statsRes.ok) {
          const statsData = await statsRes.json()
          console.log('[Privacy Preserver] Stats fetched:', statsData)
          setStats(statsData)
        } else {
          console.error('Failed to fetch stats:', statsRes.status, statsRes.statusText)
        }

        // Fetch logs
        const logsRes = await fetch(`/api/admin/privacy-preserver/logs?limit=50&_t=${timestamp}`, {
          cache: 'no-store',
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          }
        })
        if (logsRes.ok) {
          const logsData = await logsRes.json()
          console.log(`[Privacy Preserver] Logs fetched: ${logsData.length} entries`)
          setLogs(logsData)
        } else {
          console.error('Failed to fetch logs:', logsRes.status, logsRes.statusText)
        }
      } catch (err) {
        console.error('Failed to fetch Privacy Preserver data:', err)
      } finally {
        setIsLoading(false)
      }
    }

    console.log('[Privacy Preserver] Setting up data refresh')
    fetchData()
    const interval = setInterval(() => {
      console.log('[Privacy Preserver] Refreshing data...')
      fetchData()
    }, 5000) // Refresh every 5 seconds for better updates
    return () => {
      console.log('[Privacy Preserver] Cleaning up interval')
      clearInterval(interval)
    }
  }, [])

  // Prepare chart data
  const piiTypeData = stats ? Object.entries(stats.pii_by_type).map(([type, count]) => ({
    name: type.toUpperCase(),
    value: count
  })) : []

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center py-12">Loading Privacy Preserver Dashboard...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <Link href="/admin" className="inline-flex items-center text-blue-600 hover:text-blue-800 mb-4">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Admin Dashboard
          </Link>
          <div className="flex items-center gap-3">
            <Lock className="h-8 w-8 text-blue-500" />
            <h1 className="text-3xl font-bold text-gray-900">Privacy Preserver Dashboard</h1>
          </div>
          <p className="text-gray-600 mt-2">PII detection, masking, and privacy protection statistics</p>
        </div>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total PII Detections</p>
                <p className="text-2xl font-bold text-gray-900">{stats?.total_pii_detections || 0}</p>
              </div>
              <Eye className="h-8 w-8 text-orange-500" />
            </div>
            <p className="text-xs text-gray-500 mt-2">
              {stats?.pii_detection_rate.toFixed(1) || 0}% of all messages
            </p>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Masked Messages</p>
                <p className="text-2xl font-bold text-gray-900">{stats?.masked_messages || 0}</p>
              </div>
              <Lock className="h-8 w-8 text-blue-500" />
            </div>
            <p className="text-xs text-gray-500 mt-2">Messages with PII masked</p>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Recent Detections</p>
                <p className="text-2xl font-bold text-gray-900">{stats?.recent_pii_detections || 0}</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-yellow-500" />
            </div>
            <p className="text-xs text-gray-500 mt-2">Last 24 hours</p>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">PII Types Detected</p>
                <p className="text-2xl font-bold text-gray-900">{Object.keys(stats?.pii_by_type || {}).length}</p>
              </div>
              <Shield className="h-8 w-8 text-green-500" />
            </div>
            <p className="text-xs text-gray-500 mt-2">Different PII categories</p>
          </div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* PII Types Distribution */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">PII Types Distribution</h2>
            {piiTypeData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={piiTypeData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={(entry: unknown) => {
                      const e = entry as { name: string; percent: number };
                      return `${e.name}: ${(e.percent * 100).toFixed(0)}%`;
                    }}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {piiTypeData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-center py-12 text-gray-500">No PII detections yet</div>
            )}
          </div>

          {/* PII Types Bar Chart */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">PII Detections by Type</h2>
            {piiTypeData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={piiTypeData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-center py-12 text-gray-500">No PII detections yet</div>
            )}
          </div>
        </div>

        {/* PII Detection Log */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Recent PII Detections</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Timestamp
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Original Content
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Masked Content
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    PII Types
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {logs.length > 0 ? (
                  logs.map((log) => {
                    const piiTypes = log.pii_mappings 
                      ? Object.values(log.pii_mappings).map((m: { type: string }) => m.type).filter((t, i, arr) => arr.indexOf(t) === i)
                      : []
                    
                    return (
                      <tr key={log.id}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {new Date(log.timestamp).toLocaleString()}
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-900 max-w-md truncate">
                          {log.original_content}
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-600 max-w-md truncate">
                          {log.masked_content}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex flex-wrap gap-1">
                            {piiTypes.map((type, idx) => (
                              <span
                                key={idx}
                                className="px-2 py-1 text-xs font-medium bg-orange-100 text-orange-800 rounded"
                              >
                                {type}
                              </span>
                            ))}
                          </div>
                        </td>
                      </tr>
                    )
                  })
                ) : (
                  <tr>
                    <td colSpan={4} className="px-6 py-4 text-center text-gray-500">
                      No PII detections found
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

