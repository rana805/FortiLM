'use client'

import { useState, useEffect } from 'react'
import { AlertTriangle, Shield, Eye, Filter, ArrowLeft } from 'lucide-react'
import Link from 'next/link'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts'

interface FilteredResponseLog {
  id: string
  timestamp: string
  original_response: string
  filtered_response: string
  filter_analysis: Record<string, unknown> | null
  toxicity_score: number | null
  bias_detected: boolean
  bias_score: number | null
  jailbreak_detected: boolean
  jailbreak_score: number | null
  sanitization_strategy: string | null
  conversation_id: string
  explanation: string | null
}

interface OutputFilterStats {
  total_filtered: number
  toxicity_detections: number
  bias_detections: number
  jailbreak_in_output: number
  severity_distribution: Record<string, number>
  sanitization_distribution: Record<string, number>
  recent_filtered: number
  filter_rate: number
  total_responses: number
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8']

export default function OutputFilterDashboard() {
  const [stats, setStats] = useState<OutputFilterStats | null>(null)
  const [logs, setLogs] = useState<FilteredResponseLog[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Add timestamp to force fresh request
        const timestamp = Date.now()
        console.log('[Output Filter] Fetching data...', new Date().toLocaleTimeString())
        
        // Fetch stats
        const statsRes = await fetch(`/api/admin/output-filter/stats?_t=${timestamp}`, {
          cache: 'no-store',
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          }
        })
        if (statsRes.ok) {
          const statsData = await statsRes.json()
          setStats(statsData)
          console.log('[Output Filter] Stats updated:', statsData)
        } else {
          console.error('[Output Filter] Stats fetch failed:', statsRes.status, statsRes.statusText)
        }

        // Fetch logs
        const logsRes = await fetch(`/api/admin/output-filter/logs?limit=50&_t=${timestamp}`, {
          cache: 'no-store',
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          }
        })
        if (logsRes.ok) {
          const logsData = await logsRes.json()
          setLogs(logsData)
          console.log(`[Output Filter] Logs updated: ${logsData.length} messages at ${new Date().toLocaleTimeString()}`)
        } else {
          console.error('[Output Filter] Logs fetch failed:', logsRes.status, logsRes.statusText)
        }
      } catch (err) {
        console.error('[Output Filter] Failed to fetch data:', err)
      } finally {
        setIsLoading(false)
      }
    }

    console.log('[Output Filter] Setting up refresh intervals')
    fetchData()
    const interval = setInterval(() => {
      console.log('[Output Filter] Refreshing data...')
      fetchData()
    }, 5000) // Refresh every 5 seconds
    return () => clearInterval(interval)
  }, [])

  // Prepare chart data
  const filterReasonsData = stats ? [
    { name: 'Toxicity', value: stats.toxicity_detections },
    { name: 'Bias', value: stats.bias_detections },
    { name: 'Jailbreak', value: stats.jailbreak_in_output }
  ] : []

  const severityData = stats ? Object.entries(stats.severity_distribution).map(([severity, count]) => ({
    name: severity.charAt(0).toUpperCase() + severity.slice(1),
    value: count
  })) : []

  const sanitizationData = stats ? Object.entries(stats.sanitization_distribution).map(([strategy, count]) => ({
    name: strategy.charAt(0).toUpperCase() + strategy.slice(1),
    value: count
  })) : []

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center py-12">Loading Output Filter Dashboard...</div>
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
            <Filter className="h-8 w-8 text-green-500" />
            <h1 className="text-3xl font-bold text-gray-900">Output Filter Dashboard</h1>
          </div>
          <p className="text-gray-600 mt-2">Toxicity, bias, and jailbreak detection in LLM responses</p>
        </div>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Filtered</p>
                <p className="text-2xl font-bold text-gray-900">{stats?.total_filtered || 0}</p>
              </div>
              <Filter className="h-8 w-8 text-green-500" />
            </div>
            <p className="text-xs text-gray-500 mt-2">
              {stats?.filter_rate.toFixed(1) || 0}% of all responses
            </p>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Toxicity Detections</p>
                <p className="text-2xl font-bold text-gray-900">{stats?.toxicity_detections || 0}</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-red-500" />
            </div>
            <p className="text-xs text-gray-500 mt-2">Toxic content detected</p>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Bias Detections</p>
                <p className="text-2xl font-bold text-gray-900">{stats?.bias_detections || 0}</p>
              </div>
              <Eye className="h-8 w-8 text-purple-500" />
            </div>
            <p className="text-xs text-gray-500 mt-2">Biased content detected</p>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Jailbreak in Output</p>
                <p className="text-2xl font-bold text-gray-900">{stats?.jailbreak_in_output || 0}</p>
              </div>
              <Shield className="h-8 w-8 text-orange-500" />
            </div>
            <p className="text-xs text-gray-500 mt-2">Successful jailbreaks detected</p>
          </div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Filter Reasons */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Filter Reasons</h2>
            {filterReasonsData.some(d => d.value > 0) ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={filterReasonsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-center py-12 text-gray-500">No filtered responses yet</div>
            )}
          </div>

          {/* Severity Distribution */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Severity Distribution</h2>
            {severityData.some(d => d.value > 0) ? (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={severityData}
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
                    {severityData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-center py-12 text-gray-500">No severity data yet</div>
            )}
          </div>
        </div>

        {/* Sanitization Strategy */}
        <div className="bg-white rounded-lg shadow p-6 mb-8">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Sanitization Strategies</h2>
          {sanitizationData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={sanitizationData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#00C49F" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center py-12 text-gray-500">No sanitization data yet</div>
          )}
        </div>

        {/* Filtered Responses Log */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Recent Filtered Responses</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Timestamp
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Original Response
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Filtered Response
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Filter Reasons
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Strategy
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {logs.length > 0 ? (
                  logs.map((log) => {
                    const reasons = []
                    if (log.toxicity_score && log.toxicity_score > 0) reasons.push('Toxicity')
                    if (log.bias_detected) reasons.push('Bias')
                    if (log.jailbreak_detected) reasons.push('Jailbreak')
                    
                    return (
                      <tr key={log.id}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {new Date(log.timestamp).toLocaleString()}
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-900 max-w-xs truncate">
                          {log.original_response}
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-600 max-w-xs truncate">
                          {log.filtered_response || '(Blocked)'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex flex-wrap gap-1">
                            {reasons.map((reason, idx) => (
                              <span
                                key={idx}
                                className={`px-2 py-1 text-xs font-medium rounded ${
                                  reason === 'Toxicity' ? 'bg-red-100 text-red-800' :
                                  reason === 'Bias' ? 'bg-purple-100 text-purple-800' :
                                  'bg-orange-100 text-orange-800'
                                }`}
                              >
                                {reason}
                              </span>
                            ))}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {log.sanitization_strategy || 'N/A'}
                        </td>
                      </tr>
                    )
                  })
                ) : (
                  <tr>
                    <td colSpan={5} className="px-6 py-4 text-center text-gray-500">
                      No filtered responses found
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



