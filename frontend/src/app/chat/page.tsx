'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Shield, AlertTriangle, CheckCircle, Eye, Brain } from 'lucide-react'

interface Message {
  id: string
  content: string
  role: 'user' | 'assistant'
  timestamp: Date
  isFlagged?: boolean
  jailbreakDetected?: boolean
  piiDetected?: boolean
  toxicityDetected?: boolean
  explanation?: string
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Welcome to FortiLM! I\'m your secure AI assistant. I can help you with various tasks while ensuring your conversations are safe and private.',
      role: 'assistant',
      timestamp: new Date(),
      isFlagged: false
    }
  ])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendMessage = async (message: string) => {
    let timeoutId: NodeJS.Timeout | null = null
    try {
      // Use Next.js API route proxy to avoid CORS issues
      const endpoint = '/api/chat'
      console.log('Sending request to:', endpoint)
      
      // Create an AbortController for timeout
      const controller = new AbortController()
      timeoutId = setTimeout(() => {
        console.log('Request timeout triggered, aborting...')
        controller.abort()
      }, 30000) // 30 second timeout
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          message,
          conversation_id: currentConversationId 
        }),
        signal: controller.signal,
      })
      
      // Clear timeout on successful response
      if (timeoutId) {
        clearTimeout(timeoutId)
        timeoutId = null
      }
      
      console.log('Response status:', response.status)
      
      if (!response.ok) {
        const errorText = await response.text()
        console.error('Error response:', errorText)
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`)
      }
      
      const data = await response.json()
      console.log('Response data:', data)
      return data
    } catch (error) {
      // Clear timeout on error
      if (timeoutId) {
        clearTimeout(timeoutId)
      }
      
      console.error('Error sending message:', error)
      if (error instanceof Error && error.name === 'AbortError') {
        return { 
          message: "Sorry, the request timed out. The server might be taking too long to respond. Please try again.",
          is_flagged: false,
          explanation: "Request timeout after 30 seconds"
        }
      }
      return { 
        message: "Sorry, I encountered an error. Please try again.",
        is_flagged: false,
        explanation: `Network error occurred: ${error instanceof Error ? error.message : 'Unknown error'}`
      }
    }
  }

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      role: 'user',
      timestamp: new Date(),
    }

    // Add user message immediately
    setMessages(prev => [...prev, userMessage])
    const messageToSend = inputMessage
    setInputMessage('')
    setIsLoading(true)

    try {
      // Send message to the advanced chat API
      const response = await sendMessage(messageToSend)
      
      if (response.is_flagged) {
        // If response is flagged, show warning message
        const warningMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: response.message,
          role: 'assistant',
          timestamp: new Date(),
          isFlagged: true,
          jailbreakDetected: response.jailbreak_detected,
          piiDetected: response.pii_detected,
          toxicityDetected: response.toxicity_detected,
          explanation: response.explanation
        }
        
        setMessages(prev => {
          // Update the user message with PII detection info if detected
          const updated = [...prev]
          const lastUserMsgIndex = updated.length - 1
          if (response.pii_detected && updated[lastUserMsgIndex] && updated[lastUserMsgIndex].role === 'user') {
            updated[lastUserMsgIndex] = {
              ...updated[lastUserMsgIndex],
              piiDetected: true,
              explanation: response.explanation || "PII detected and masked"
            }
          }
          return [...updated, warningMessage]
        })
      } else {
        // Normal AI response (may still have PII detected but not flagged)
        const assistantMessage: Message = {
          id: (Date.now() + 2).toString(),
          content: response.message,
          role: 'assistant',
          timestamp: new Date(),
          isFlagged: response.is_flagged,
          jailbreakDetected: response.jailbreak_detected,
          piiDetected: response.pii_detected,
          toxicityDetected: response.toxicity_detected,
          explanation: response.explanation
        }

        setMessages(prev => {
          // Update the user message with PII detection info if detected
          const updated = [...prev]
          const lastUserMsgIndex = updated.length - 1
          if (response.pii_detected && updated[lastUserMsgIndex] && updated[lastUserMsgIndex].role === 'user') {
            updated[lastUserMsgIndex] = {
              ...updated[lastUserMsgIndex],
              piiDetected: true,
              explanation: response.explanation || "PII detected and masked"
            }
          }
          return [...updated, assistantMessage]
        })
        
        // Update conversation ID if this is a new conversation
        if (response.conversation_id && !currentConversationId) {
          setCurrentConversationId(response.conversation_id)
        }
      }
    } catch (error) {
      console.error('Error in chat:', error)
      const errorMessage: Message = {
        id: (Date.now() + 3).toString(),
        content: 'Sorry, I encountered an error. Please try again.',
        role: 'assistant',
        timestamp: new Date(),
        isFlagged: false
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const getSecurityIcon = (message: Message) => {
    // Show PII indicator even if not flagged (PII is masked, not blocked)
    if (message.piiDetected) {
      return <span title="PII detected and masked"><Eye className="h-4 w-4 text-orange-500" /></span>
    }
    if (message.isFlagged) {
      if (message.jailbreakDetected) {
        return <span title="Jailbreak detected"><Shield className="h-4 w-4 text-red-500" /></span>
      }
      if (message.toxicityDetected) {
        return <span title="Toxic content detected"><AlertTriangle className="h-4 w-4 text-yellow-500" /></span>
      }
      return <span title="Content flagged"><AlertTriangle className="h-4 w-4 text-red-500" /></span>
    }
    return <span title="Content safe"><CheckCircle className="h-4 w-4 text-green-500" /></span>
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Shield className="h-6 w-6 text-blue-600 mr-2" />
              <h1 className="text-xl font-semibold text-gray-900">FortiLM Chat</h1>
            </div>
            <div className="flex items-center text-sm text-gray-500">
              <Brain className="h-4 w-4 mr-1" />
              Secure AI Assistant
            </div>
          </div>
        </div>
      </header>

      {/* Chat Container */}
      <div className="max-w-4xl mx-auto h-[calc(100vh-140px)] flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-lg px-4 py-2 ${
                  message.role === 'user'
                    ? message.piiDetected
                      ? 'bg-blue-600 text-white border-2 border-orange-400'
                      : 'bg-blue-600 text-white'
                    : message.isFlagged
                    ? 'bg-red-50 border border-red-200 text-red-800'
                    : message.piiDetected
                    ? 'bg-orange-50 border border-orange-200 text-gray-900'
                    : 'bg-white border border-gray-200 text-gray-900'
                }`}
              >
                <div className="flex items-start justify-between mb-1">
                  <div className="flex items-center">
                    {getSecurityIcon(message)}
                    <span className="ml-2 text-xs font-medium">
                      {message.role === 'user' ? 'You' : 'FortiLM'}
                    </span>
                  </div>
                  <span className="text-xs opacity-70" suppressHydrationWarning>
                    {message.timestamp.toISOString().slice(11, 19)}
                  </span>
                </div>
                <p className="text-sm">{message.content}</p>
                {message.explanation && (
                  <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded text-xs text-gray-900">
                    <strong>Security Analysis:</strong> {message.explanation}
                  </div>
                )}
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 rounded-lg px-4 py-2">
                <div className="flex items-center">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                  <span className="text-sm text-gray-600">FortiLM is thinking...</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t bg-white p-4">
          <div className="flex space-x-2">
            <div className="flex-1">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message here... (Press Enter to send, Shift+Enter for new line)"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none bg-white text-gray-900 placeholder-gray-500"
                rows={2}
                disabled={isLoading}
              />
            </div>
            <button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
          
          {/* Security Status */}
          <div className="mt-2 flex items-center text-xs text-gray-500">
            <Shield className="h-3 w-3 mr-1" />
            <span>All messages are analyzed for security threats before processing</span>
          </div>
        </div>
      </div>
    </div>
  )
}
