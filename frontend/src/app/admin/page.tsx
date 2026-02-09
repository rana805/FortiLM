'use client'

import { useState, useEffect } from 'react'
import { Shield, Users, AlertTriangle, BarChart3, Eye, Brain, Zap, Lock, Filter, TrendingUp } from 'lucide-react'
import Link from 'next/link'

interface SecurityStats {
  totalMessages: number
  flaggedMessages: number
  jailbreakAttempts: number
  piiDetected: number
  toxicityDetected: number
  blockedRequests: number
}

interface RecentActivity {
  id: string
  timestamp: Date
  type: 'jailbreak' | 'pii' | 'toxicity' | 'normal'
  message: string
  severity: 'low' | 'medium' | 'high'
}

interface BenchmarkResult {
  raw_ms: number[]
  fortilm_ms: number[]
  raw_avg_ms: number
  fortilm_avg_ms: number
  raw_p50_ms: number
  raw_p95_ms: number
  fortilm_p50_ms: number
  fortilm_p95_ms: number
}

interface Metrics {
  active_users: number
  system_status: 'Operational' | 'Degraded' | string
  recent_errors: number
}

export default function AdminDashboard() {
  const [stats, setStats] = useState<SecurityStats>({
    totalMessages: 0,
    flaggedMessages: 0,
    jailbreakAttempts: 0,
    piiDetected: 0,
    toxicityDetected: 0,
    blockedRequests: 0
  })

  const [recentActivity, setRecentActivity] = useState<RecentActivity[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isBenchmarking, setIsBenchmarking] = useState(false)
  const [benchmark, setBenchmark] = useState<BenchmarkResult | null>(null)
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<Metrics>({
    active_users: 0,
    system_status: 'Unknown',
    recent_errors: 0
  })

  useEffect(() => {
    const fetchActivity = async () => {
      try {
        // Add timestamp to force fresh request
        const timestamp = Date.now()
        const res = await fetch(`/api/chat-stats/recent-activity?_t=${timestamp}`, {
          cache: 'no-store',
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          }
        })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        const mapped: RecentActivity[] = data.map((e: { id: string; timestamp: string; type: string; message: string; severity: string }) => {
          // Fix timestamp parsing - handle invalid dates gracefully
          let timestamp: Date
          if (e.timestamp) {
            // Remove duplicate Z if present with timezone offset
            const cleanTimestamp = e.timestamp.replace(/([+-]\d{2}:\d{2})Z$/, '$1')
            timestamp = new Date(cleanTimestamp)
            // If still invalid, use current time
            if (isNaN(timestamp.getTime())) {
              timestamp = new Date()
            }
          } else {
            timestamp = new Date()
          }
          return {
            id: e.id,
            timestamp,
            type: (e.type as 'jailbreak' | 'pii' | 'toxicity' | 'normal') ?? 'normal',
            message: e.message,
            severity: (e.severity as 'low' | 'medium' | 'high') ?? 'low',
          }
        })
        // Always update state - React will handle re-rendering efficiently
        setRecentActivity(mapped)
        console.log(`[Admin Dashboard] Updated ${mapped.length} activities at ${new Date().toLocaleTimeString()}`)
        setIsLoading(false)
      } catch (err) {
        // keep previous data; show loading false after first attempt
        setIsLoading(false)
      }
    }

    const fetchDashboardStats = async () => {
      try {
        // Add timestamp to force fresh request
        const timestamp = Date.now()
        const res = await fetch(`/api/admin/dashboard-stats?_t=${timestamp}`, {
          cache: 'no-store',
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          }
        })
        if (res.ok) {
          const data = await res.json()
          // Count flagged from recent activity (for display purposes)
          const flagged = recentActivity.filter(a => a.type !== 'normal').length
          
          setStats({
            totalMessages: data.total_messages || 0,
            flaggedMessages: flagged,
            jailbreakAttempts: data.jailbreak_attempts || 0,
            piiDetected: data.recent_pii_detections || 0, // Use recent PII (last 24h) for dashboard
            toxicityDetected: data.toxicity_detected || 0,
            blockedRequests: flagged,
          })
          console.log(`[Admin Dashboard] Stats updated: ${data.total_messages} total messages, ${data.recent_pii_detections} recent PII`)
        }
      } catch (err) {
        console.error('[Admin Dashboard] Failed to fetch dashboard stats:', err)
      }
    }

    const fetchMetrics = async () => {
      try {
        // Add timestamp to force fresh request
        const timestamp = Date.now()
        const res = await fetch(`/api/chat-stats/metrics?_t=${timestamp}`, {
          cache: 'no-store',
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          }
        })
        if (res.ok) {
          const data = await res.json()
          setMetrics(data as Metrics)
          console.log('[Admin Dashboard] Metrics fetched:', data)
        } else {
          console.error('[Admin Dashboard] Failed to fetch metrics:', res.status, res.statusText)
          // Set default status if fetch fails
          setMetrics({
            system_status: 'Unknown',
            recent_errors: 0
          })
        }
      } catch (err) {
        console.error('[Admin Dashboard] Error fetching metrics:', err)
        // Set default status on error
        setMetrics({
          system_status: 'Unknown',
          recent_errors: 0
        })
      }
    }

    console.log('[Admin Dashboard] Setting up refresh intervals')
    fetchActivity(); fetchDashboardStats(); fetchMetrics()
    const id1 = setInterval(() => {
      console.log('[Admin Dashboard] Refreshing activity...')
      fetchActivity()
    }, 5000)
    const id2 = setInterval(() => {
      console.log('[Admin Dashboard] Refreshing dashboard stats...')
      fetchDashboardStats()
    }, 5000)
    const id3 = setInterval(() => {
      console.log('[Admin Dashboard] Refreshing metrics...')
      fetchMetrics()
    }, 5000)
    return () => { 
      console.log('[Admin Dashboard] Cleaning up intervals')
      clearInterval(id1); 
      clearInterval(id2);
      clearInterval(id3)
    }
  }, [])

  const runBenchmark = async () => {
    try {
      setIsBenchmarking(true)
      setBenchmarkError(null)
      const res = await fetch(`/api/chat-stats/benchmark`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: 'Summarize FortiLM in one sentence.', iterations: 3 })
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setBenchmark(data as BenchmarkResult)
    } catch (e: unknown) {
      setBenchmarkError(e instanceof Error ? e.message : 'Benchmark failed')
    } finally {
      setIsBenchmarking(false)
    }
  }

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'jailbreak':
        return <Shield className="h-4 w-4 text-red-500" />
      case 'pii':
        return <Eye className="h-4 w-4 text-orange-500" />
      case 'toxicity':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />
      default:
        return <Brain className="h-4 w-4 text-green-500" />
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high':
        return 'bg-red-100 text-red-800 border-red-200'
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      default:
        return 'bg-green-100 text-green-800 border-green-200'
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading admin dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Shield className="h-6 w-6 text-blue-600 mr-2" />
              <h1 className="text-xl font-semibold text-gray-900">FortiLM Admin Dashboard</h1>
            </div>
            <div className="flex items-center text-sm text-gray-500">
              <BarChart3 className="h-4 w-4 mr-1" />
              Security Monitoring
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Quick Actions - Iteration 2 Dashboards */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Iteration 2 Dashboards</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Link href="/admin/privacy-preserver" className="bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg shadow p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-center gap-3">
                <Lock className="h-6 w-6" />
                <div>
                  <h3 className="font-bold text-lg">Privacy Preserver</h3>
                  <p className="text-sm text-blue-100">PII detection & masking stats</p>
                </div>
              </div>
            </Link>
            <Link href="/admin/output-filter" className="bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg shadow p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-center gap-3">
                <Filter className="h-6 w-6" />
                <div>
                  <h3 className="font-bold text-lg">Output Filter</h3>
                  <p className="text-sm text-green-100">Toxicity, bias & jailbreak stats</p>
                </div>
              </div>
            </Link>
            <Link href="/admin/unified-security" className="bg-gradient-to-r from-indigo-500 to-indigo-600 text-white rounded-lg shadow p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-center gap-3">
                <TrendingUp className="h-6 w-6" />
                <div>
                  <h3 className="font-bold text-lg">Unified Overview</h3>
                  <p className="text-sm text-indigo-100">Complete security dashboard</p>
                </div>
              </div>
            </Link>
          </div>
        </div>

        {/* Benchmark */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold text-gray-900">LLM Benchmark</h2>
            <button
              onClick={runBenchmark}
              disabled={isBenchmarking}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              {isBenchmarking ? 'Running…' : 'Run benchmark'}
            </button>
          </div>
          {benchmarkError && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded text-sm">{benchmarkError}</div>
          )}
          {benchmark && (
            <div className="bg-white rounded-lg shadow p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Raw (direct Groq)</h3>
                  <p className="text-sm text-gray-600">avg: {benchmark.raw_avg_ms.toFixed(1)} ms</p>
                  <p className="text-sm text-gray-600">p50: {benchmark.raw_p50_ms.toFixed(1)} ms • p95: {benchmark.raw_p95_ms.toFixed(1)} ms</p>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">FortiLM pipeline</h3>
                  <p className="text-sm text-gray-600">avg: {benchmark.fortilm_avg_ms.toFixed(1)} ms</p>
                  <p className="text-sm text-gray-600">p50: {benchmark.fortilm_p50_ms.toFixed(1)} ms • p95: {benchmark.fortilm_p95_ms.toFixed(1)} ms</p>
                </div>
              </div>
            </div>
          )}
        </div>
        {/* Security Overview */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Security Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <BarChart3 className="h-6 w-6 text-blue-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Total Messages</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.totalMessages.toLocaleString()}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className="p-2 bg-red-100 rounded-lg">
                  <AlertTriangle className="h-6 w-6 text-red-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Flagged Messages</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.flaggedMessages}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className="p-2 bg-green-100 rounded-lg">
                  <Shield className="h-6 w-6 text-green-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Block Rate</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {((stats.flaggedMessages / stats.totalMessages) * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <Link href="/admin/privacy-preserver" className="bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg shadow p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-center gap-3">
                <Lock className="h-6 w-6" />
                <div>
                  <h3 className="font-bold text-lg">Privacy Preserver</h3>
                  <p className="text-sm text-blue-100">View PII detection & masking stats</p>
                </div>
              </div>
            </Link>
            <Link href="/admin/output-filter" className="bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg shadow p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-center gap-3">
                <Filter className="h-6 w-6" />
                <div>
                  <h3 className="font-bold text-lg">Output Filter</h3>
                  <p className="text-sm text-green-100">View toxicity, bias & jailbreak stats</p>
                </div>
              </div>
            </Link>
            <Link href="/admin/unified-security" className="bg-gradient-to-r from-indigo-500 to-indigo-600 text-white rounded-lg shadow p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-center gap-3">
                <TrendingUp className="h-6 w-6" />
                <div>
                  <h3 className="font-bold text-lg">Unified Overview</h3>
                  <p className="text-sm text-indigo-100">Complete security dashboard</p>
                </div>
              </div>
            </Link>
          </div>
        </div>

        {/* Security Modules Status */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Security Modules</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Shield className="h-5 w-5 text-red-500 mr-2" />
                  <span className="font-medium">Jailbreak Detection</span>
                </div>
                <span className="text-sm text-gray-900">{stats.jailbreakAttempts} blocked</span>
              </div>
              <div className="mt-2">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-red-500 h-2 rounded-full transition-all duration-700"
                    style={{ width: `${Math.min(100, stats.totalMessages ? (stats.jailbreakAttempts / stats.totalMessages) * 100 : 0)}%` }}
                  ></div>
                </div>
              </div>
            </div>

            <Link href="/admin/privacy-preserver" className="bg-white rounded-lg shadow p-4 hover:shadow-lg transition-shadow block">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Eye className="h-5 w-5 text-orange-500 mr-2" />
                  <span className="font-medium">Privacy Preserver</span>
                </div>
                <span className="text-sm text-gray-900">{stats.piiDetected} detected</span>
              </div>
              <div className="mt-2">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-orange-500 h-2 rounded-full transition-all duration-700"
                    style={{ width: `${Math.min(100, stats.totalMessages ? (stats.piiDetected / stats.totalMessages) * 100 : 0)}%` }}
                  ></div>
                </div>
              </div>
              <p className="text-xs text-blue-600 mt-2">View detailed dashboard →</p>
            </Link>

            <Link href="/admin/output-filter" className="bg-white rounded-lg shadow p-4 hover:shadow-lg transition-shadow block">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <AlertTriangle className="h-5 w-5 text-yellow-500 mr-2" />
                  <span className="font-medium">Output Filter</span>
                </div>
                <span className="text-sm text-gray-900">{stats.toxicityDetected} filtered</span>
              </div>
              <div className="mt-2">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-yellow-500 h-2 rounded-full transition-all duration-700"
                    style={{ width: `${Math.min(100, stats.totalMessages ? (stats.toxicityDetected / stats.totalMessages) * 100 : 0)}%` }}
                  ></div>
                </div>
              </div>
            </Link>

            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Brain className="h-5 w-5 text-purple-500 mr-2" />
                  <span className="font-medium">Explainability</span>
                </div>
                <span className="text-sm text-gray-500">Active</span>
              </div>
              <div className="mt-2">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-purple-500 h-2 rounded-full" style={{ width: '100%' }}></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Recent Security Activity</h2>
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Time
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Severity
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Message
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {recentActivity.length === 0 ? (
                    <tr>
                      <td colSpan={4} className="px-6 py-8 text-center text-sm text-gray-500">
                        No recent activity. Send a message through the chat to see activity here.
                      </td>
                    </tr>
                  ) : (
                    recentActivity.map((activity) => (
                      <tr key={activity.id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {activity.timestamp && !isNaN(activity.timestamp.getTime()) 
                            ? activity.timestamp.toLocaleTimeString() 
                            : 'Invalid Date'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            {getActivityIcon(activity.type)}
                            <span className="ml-2 text-sm font-medium text-gray-900 capitalize">
                              {activity.type}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full border ${getSeverityColor(activity.severity)}`}>
                            {activity.severity}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-900 max-w-xs truncate">
                          {activity.message}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* System Status */}
        <div>
          <h2 className="text-2xl font-bold text-gray-900 mb-6">System Status</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className="p-2 bg-green-100 rounded-lg">
                  <Zap className="h-6 w-6 text-green-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">API Status</p>
                  <p className={`text-lg font-bold ${
                    metrics.system_status === 'Operational' ? 'text-green-600' : 
                    metrics.system_status === 'Degraded' ? 'text-yellow-600' : 
                    'text-red-600'
                  }`}>
                    {metrics.system_status || 'Unknown'}
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className="p-2 bg-green-100 rounded-lg">
                  <Lock className="h-6 w-6 text-green-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Database</p>
                  <p className="text-lg font-bold text-green-600">Connected</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className="p-2 bg-green-100 rounded-lg">
                  <Users className="h-6 w-6 text-green-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Active Users</p>
                  <p className="text-lg font-bold text-gray-900">{metrics?.active_users ?? 0}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
