import { Shield, Lock, Eye, Brain, Users, BarChart3, Zap, Plug } from 'lucide-react'
import Link from 'next/link'

export default function HomePage() {
  const modules = [
    {
      title: "Hybrid Jailbreak Detection",
      description: "Rule-based + AI detection for adversarial prompts",
      icon: Shield,
      color: "text-red-500"
    },
    {
      title: "Privacy Preserver",
      description: "PII detection and masking for data protection",
      icon: Lock,
      color: "text-blue-500"
    },
    {
      title: "Output Filter",
      description: "Content safety filtering and toxicity detection",
      icon: Eye,
      color: "text-green-500"
    },
    {
      title: "Explainability",
      description: "Transparent decision making with SHAP/LIME",
      icon: Brain,
      color: "text-purple-500"
    },
    {
      title: "Adaptive Learning",
      description: "Continuous improvement from feedback",
      icon: Zap,
      color: "text-yellow-500"
    },
    {
      title: "Admin Dashboard",
      description: "System monitoring and management",
      icon: BarChart3,
      color: "text-indigo-500"
    },
    {
      title: "Web UI",
      description: "Clean and user-friendly chat interface",
      icon: Users,
      color: "text-pink-500"
    },
    {
      title: "Plugin Integration",
      description: "Middleware for existing chatbot systems",
      icon: Plug,
      color: "text-orange-500"
    }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <Shield className="h-8 w-8 text-blue-600 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900">FortiLM</h1>
            </div>
            <div className="flex space-x-4">
              <Link href="/auth/login">
                <button className="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50">Login</button>
              </Link>
              <Link href="/auth/register">
                <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">Get Started</button>
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            Security Middleware for
            <span className="text-blue-600"> Large Language Models</span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            FortiLM provides comprehensive defense against jailbreak attacks, prompt injections, 
            unsafe outputs, and data privacy risks in AI systems.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/chat">
              <button className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 w-full sm:w-auto">
                Start Chatting Securely
              </button>
            </Link>
                   <Link href="/admin">
                     <button className="px-6 py-3 border border-gray-300 rounded-lg hover:bg-gray-50 w-full sm:w-auto text-gray-700 font-medium">
                       Admin Dashboard
                     </button>
                   </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Comprehensive Security Modules
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Our 8-module architecture provides multi-layered protection for your AI applications
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {modules.map((module, index) => (
              <div key={index} className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow border border-gray-200">
                <div className="flex items-center mb-2">
                  <module.icon className={`h-6 w-6 ${module.color} mr-2`} />
                  <h3 className="text-lg font-semibold text-gray-900">{module.title}</h3>
                </div>
                <p className="text-gray-600 text-sm">{module.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Why Choose FortiLM?
            </h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-blue-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <Shield className="h-8 w-8 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Enterprise-Grade Security</h3>
              <p className="text-gray-600">
                Multi-layered protection against jailbreak attacks, prompt injections, and data leaks
              </p>
            </div>
            
            <div className="text-center">
              <div className="bg-green-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <Brain className="h-8 w-8 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Explainable AI</h3>
              <p className="text-gray-600">
                Transparent decision-making with SHAP/LIME explanations for every security action
              </p>
            </div>
            
            <div className="text-center">
              <div className="bg-purple-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <Plug className="h-8 w-8 text-purple-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Easy Integration</h3>
              <p className="text-gray-600">
                Plugin-style middleware that integrates seamlessly with existing chatbot systems
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="flex items-center justify-center mb-4">
              <Shield className="h-6 w-6 text-blue-400 mr-2" />
              <span className="text-xl font-bold">FortiLM</span>
            </div>
            <p className="text-gray-400 mb-4">
              Security Middleware for Large Language Models
            </p>
            <p className="text-sm text-gray-500">
              © 2024 National University of Computer and Emerging Sciences. FYP Project.
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}