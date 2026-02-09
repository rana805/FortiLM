'use client'

import { useState, useEffect } from 'react'
import { Shield, Lock, Eye, Filter, AlertTriangle, TrendingUp, ArrowLeft } from 'lucide-react'
import Link from 'next/link'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'

interface UnifiedSecurityStats {
  total_conversations: number
  flagged_conversations: number
  security_health_score: number
  module_stats: {
    jailbreak_prompt: number
    pii_detected: number
    toxicity_detected: number
    bias_detected: number
    jailbreak_output: number
  }
  recent_activity: {
    conversations: number
    flagged: number
  }
  threat_trends: Array<{
    date: string
    threats: number
  }>
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d']

export default function UnifiedSecurityDashboard() {
  const [stats, setStats] = useState<UnifiedSecurityStats | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Add timestamp to force fresh request
        const timestamp = Date.now()
        const res = await fetch(`/api/admin/unified-security/stats?_t=${timestamp}`, {
          cache: 'no-store',
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          }
        })
        if (res.ok) {
          const data = await res.json()
          setStats(data)
        }
      } catch (err) {
        console.error('Failed to fetch unified security data:', err)
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 5000) // Refresh every 5 seconds
    return () => clearInterval(interval)
  }, [])

  // Prepare chart data
  const moduleStatsData = stats ? [
    { name: 'Jailbreak (Prompt)', value: stats.module_stats.jailbreak_prompt },
    { name: 'PII Detected', value: stats.module_stats.pii_detected },
    { name: 'Toxicity', value: stats.module_stats.toxicity_detected },
    { name: 'Bias', value: stats.module_stats.bias_detected },
    { name: 'Jailbreak (Output)', value: stats.module_stats.jailbreak_output }
  ] : []

  const getHealthColor = (score: number) => {
    if (score >= 80) return 'text-green-600'
    if (score >= 60) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getHealthBgColor = (score: number) => {
    if (score >= 80) return 'bg-green-100'
    if (score >= 60) return 'bg-yellow-100'
    return 'bg-red-100'
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center py-12">Loading Unified Security Dashboard...</div>
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
            <Shield className="h-8 w-8 text-indigo-500" />
            <h1 className="text-3xl font-bold text-gray-900">Unified Security Overview</h1>
          </div>
          <p className="text-gray-600 mt-2">Comprehensive security metrics across all modules</p>
        </div>

        {/* Security Health Score */}
        <div className="bg-white rounded-lg shadow p-6 mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-gray-900 mb-2">Security Health Score</h2>
              <p className="text-sm text-gray-600">Overall system security status</p>
            </div>
            <div className={`text-6xl font-bold ${getHealthColor(stats?.security_health_score || 0)}`}>
              {stats?.security_health_score.toFixed(0) || 0}
            </div>
          </div>
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-4">
              <div
                className={`h-4 rounded-full transition-all duration-700 ${getHealthBgColor(stats?.security_health_score || 0)}`}
                style={{ width: `${Math.min(100, stats?.security_health_score || 0)}%` }}
              ></div>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              {stats?.security_health_score || 0 >= 80 ? 'Excellent' : 
               stats?.security_health_score || 0 >= 60 ? 'Good' : 'Needs Attention'}
            </p>
          </div>
        </div>

        {/* Overall Statistics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Conversations</p>
                <p className="text-2xl font-bold text-gray-900">{stats?.total_conversations || 0}</p>
              </div>
              <Shield className="h-8 w-8 text-blue-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Flagged Conversations</p>
                <p className="text-2xl font-bold text-gray-900">{stats?.flagged_conversations || 0}</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-red-500" />
            </div>
            <p className="text-xs text-gray-500 mt-2">
              {stats && stats.total_conversations > 0
                ? ((stats.flagged_conversations / stats.total_conversations) * 100).toFixed(1)
                : 0}% of total
            </p>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Recent Activity</p>
                <p className="text-2xl font-bold text-gray-900">{stats?.recent_activity.conversations || 0}</p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-500" />
            </div>
            <p className="text-xs text-gray-500 mt-2">Last 24 hours</p>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Recent Threats</p>
                <p className="text-2xl font-bold text-gray-900">{stats?.recent_activity.flagged || 0}</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-orange-500" />
            </div>
            <p className="text-xs text-gray-500 mt-2">Last 24 hours</p>
          </div>
        </div>

        {/* Module Statistics */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Module Stats Bar Chart */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Module Detection Statistics</h2>
            {moduleStatsData.some(d => d.value > 0) ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={moduleStatsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-center py-12 text-gray-500">No detection data yet</div>
            )}
          </div>

          {/* Module Stats Pie Chart */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Threat Distribution</h2>
            {moduleStatsData.some(d => d.value > 0) ? (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={moduleStatsData.filter(d => d.value > 0)}
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
                    {moduleStatsData.filter(d => d.value > 0).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-center py-12 text-gray-500">No threat data yet</div>
            )}
          </div>
        </div>

        {/* Threat Trends */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Threat Trends (Last 7 Days)</h2>
          {stats && stats.threat_trends.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={stats.threat_trends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="threats" stroke="#8884d8" strokeWidth={2} name="Threats" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center py-12 text-gray-500">No trend data available</div>
          )}
        </div>
      </div>
    </div>
  )
}

