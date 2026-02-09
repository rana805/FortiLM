import { NextRequest, NextResponse } from 'next/server'

// Always use 127.0.0.1 for server-side fetch (ignore NEXT_PUBLIC_API_URL which is for client-side)
// The backend should be accessible on localhost from the Next.js server
const API_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

async function proxyRequest(
  request: NextRequest,
  params: { path: string[] },
  method: 'GET' | 'POST' = 'GET'
) {
  try {
    const path = Array.isArray(params.path) ? params.path.join('/') : params.path
    const searchParams = request.nextUrl.searchParams.toString()
    const queryString = searchParams ? `?${searchParams}` : ''
    const backendUrl = `${API_URL}/api/v1/chat/${path}${queryString}`
    
    const fetchOptions: RequestInit = {
      method,
      headers: {
        'Content-Type': 'application/json',
      },
    }
    
    if (method === 'POST') {
      const body = await request.json()
      fetchOptions.body = JSON.stringify(body)
    }
    
    // Use native fetch with proper error handling
    let response: Response
    try {
      response = await fetch(backendUrl, fetchOptions)
    } catch (fetchError) {
      console.error('Fetch error:', fetchError)
      return NextResponse.json(
        { 
          error: 'Failed to connect to backend', 
          details: fetchError instanceof Error ? fetchError.message : 'Network error'
        },
        { status: 500 }
      )
    }
    
    if (!response.ok) {
      let errorText = ''
      try {
        errorText = await response.text()
      } catch {
        errorText = 'Unknown error'
      }
      return NextResponse.json(
        { error: `Backend error: ${response.status}`, details: errorText },
        { status: response.status }
      )
    }
    
    let data: any
    try {
      data = await response.json()
    } catch (jsonError) {
      console.error('JSON parse error:', jsonError)
      return NextResponse.json(
        { error: 'Invalid response from backend' },
        { status: 500 }
      )
    }
    
    const nextResponse = NextResponse.json(data)
    // Add cache-busting headers
    nextResponse.headers.set('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate')
    nextResponse.headers.set('Pragma', 'no-cache')
    nextResponse.headers.set('Expires', '0')
    return nextResponse
  } catch (error) {
    console.error('Proxy error:', error)
    return NextResponse.json(
      { 
        error: 'Internal server error', 
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    )
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  return proxyRequest(request, params, 'GET')
}

export async function POST(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  return proxyRequest(request, params, 'POST')
}

