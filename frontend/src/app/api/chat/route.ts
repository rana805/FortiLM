import { NextRequest, NextResponse } from 'next/server'

// Always use 127.0.0.1 for server-side fetch (ignore NEXT_PUBLIC_API_URL which is for client-side)
// The backend should be accessible on localhost from the Next.js server
const API_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const backendUrl = `${API_URL}/api/v1/chat`
    
    // Use native fetch with proper error handling
    let response: Response
    try {
      response = await fetch(backendUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      })
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
    
    return NextResponse.json(data)
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

