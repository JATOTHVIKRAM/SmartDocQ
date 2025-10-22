import { useState } from "react";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  ArrowRight, 
  Zap, 
  FileText, 
  MessageSquare, 
  Upload, 
  Send,
  Star,
  Mail,
  Phone,
  MapPin,
  Check,
  Play,
  User,
  Bot,
  ChevronLeft,
  ChevronRight
} from "lucide-react";
import FileUpload from "@/components/file-upload";

interface Message {
  id: string;
  type: "user" | "bot";
  content: string;
  timestamp: Date;
}

interface LandingProps {
  isAuthenticated?: boolean;
  onAuthModal?: (type: "login" | "signup") => void;
}

export default function Landing({ isAuthenticated = false, onAuthModal }: LandingProps) {
  const [, setLocation] = useLocation();
  const [currentReview, setCurrentReview] = useState(0);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      type: "user",
      content: "How do I authenticate with the API?",
      timestamp: new Date(Date.now() - 60000),
    },
    {
      id: "2",
      type: "bot",
      content: "Based on your API documentation, authentication requires an API key in the Authorization header. Here's the format: `Authorization: Bearer YOUR_API_KEY`",
      timestamp: new Date(Date.now() - 30000),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const reviews = [
    {
      id: 1,
      name: "Sarah Johnson",
      role: "Senior Developer",
      avatar: "SJ",
      rating: 5,
      text: "SmartDocQ has revolutionized how I handle technical documentation. The AI responses are incredibly accurate and save me hours of manual searching."
    },
    {
      id: 2,
      name: "Michael Chen",
      role: "HR Manager",
      avatar: "MC",
      rating: 5,
      text: "The Interview Copilot feature is a game-changer for HR. It generates thoughtful questions that help us assess candidates more effectively."
    },
    {
      id: 3,
      name: "Dr. Emily Rodriguez",
      role: "Research Analyst",
      avatar: "ER",
      rating: 5,
      text: "As a researcher, I deal with hundreds of documents daily. SmartDocQ makes it effortless to extract key information and insights."
    },
    {
      id: 4,
      name: "Alex Thompson",
      role: "Product Manager",
      avatar: "AT",
      rating: 5,
      text: "SmartDocQ has become an essential tool in our workflow. The accuracy of responses and the intuitive interface make it perfect for both technical and non-technical team members."
    },
    {
      id: 5,
      name: "Jennifer Lee",
      role: "Content Strategist",
      avatar: "JL",
      rating: 5,
      text: "The document analysis capabilities are outstanding. It helps me quickly understand complex documents and create better content strategies."
    }
  ];

  const handleSendMessage = () => {
    if (!inputValue.trim()) return;

    const newMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, newMessage]);
    setInputValue("");

    // Simulate bot response
    setTimeout(() => {
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        content: "I'm analyzing your document to provide the best answer. This is a demo response showing how SmartDocQ processes your questions.",
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, botResponse]);
    }, 1000);
  };

  const handleFileSelect = (file: File) => {
    setUploadedFile(file);
    console.log("File uploaded:", file.name);
  };

  const handleTryFeature = (feature: "smart-documentation" | "interview-copilot") => {
    if (isAuthenticated) {
      if (feature === "smart-documentation") {
        setLocation("/smart-documentation");
      } else if (feature === "interview-copilot") {
        setLocation("/interview-copilot");
      } else {
        setLocation(`/${feature}`);
      }
    } else {
      onAuthModal?.("signup");
    }
  };

  const nextReview = () => {
    setCurrentReview((prev) => (prev + 1) % reviews.length);
  };

  const prevReview = () => {
    setCurrentReview((prev) => (prev - 1 + reviews.length) % reviews.length);
  };

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center px-4 py-2 mb-8 bg-card-bg/50 border border-primary-blue/30 rounded-full">
            <Zap className="text-primary-blue mr-2 h-4 w-4" />
            <span className="text-primary-blue text-sm font-medium">AI-Powered Documentation</span>
          </div>

          <h1 className="text-4xl md:text-6xl font-bold mb-6">
            SmartDocQ
            <br />
            <span className="text-primary-blue">Your AI Assistant for</span>
            <br />
            Smarter Document Q&A
          </h1>

          <p className="text-xl text-text-secondary mb-12 max-w-2xl mx-auto">
            Transforming document search into a smart, interactive experience. Get instant answers, prepare for interviews, and unlock the power of your documents.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Button
              size="lg"
              className="bg-primary-blue hover:bg-blue-600 px-8 py-4 text-lg font-semibold transition-all transform hover:scale-105"
              onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
              data-testid="button-get-started"
            >
              Get Started Free <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
            <Button
              variant="outline"
              size="lg"
              className="border-border-color hover:border-text-secondary px-8 py-4 text-lg font-semibold transition-colors"
              onClick={() => document.getElementById('demo')?.scrollIntoView({ behavior: 'smooth' })}
              data-testid="button-watch-demo"
            >
              Watch Demo
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-4 sm:px-6 lg:px-8 bg-card-bg/20">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Our Features</h2>
            <p className="text-xl text-text-secondary">Discover how SmartDocQ transforms your document workflow</p>
          </div>

          {/* Side-by-Side Feature Cards */}
          <div className="grid md:grid-cols-2 gap-8">
            
            {/* Smart Documentation Card */}
            <div className="bg-card-bg p-8 rounded-2xl border border-border-color hover:border-primary-blue/50 transition-all">
              <div className="w-16 h-16 bg-primary-blue/20 rounded-2xl flex items-center justify-center mb-6">
                <FileText className="text-primary-blue h-8 w-8" />
              </div>
              
              <h3 className="text-2xl font-bold mb-4">Smart Documentation</h3>
              <p className="text-text-secondary mb-6">
                Transform any document into an intelligent, searchable knowledge base. Ask questions and get instant, contextual answers powered by advanced AI.
              </p>
              
              <div className="mb-8">
                <h4 className="font-semibold mb-4 text-primary-blue">How it works:</h4>
                <ul className="space-y-3">
                  <li className="flex items-start">
                    <div className="w-6 h-6 bg-primary-blue rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                      <span className="text-white text-xs font-bold">1</span>
                    </div>
                    <span className="text-sm text-text-secondary">Upload your PDF, DOCX, or TXT documents</span>
                  </li>
                  <li className="flex items-start">
                    <div className="w-6 h-6 bg-primary-blue rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                      <span className="text-white text-xs font-bold">2</span>
                    </div>
                    <span className="text-sm text-text-secondary">AI analyzes and understands your content</span>
                  </li>
                  <li className="flex items-start">
                    <div className="w-6 h-6 bg-primary-blue rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                      <span className="text-white text-xs font-bold">3</span>
                    </div>
                    <span className="text-sm text-text-secondary">Ask questions in natural language and get instant answers</span>
                  </li>
                </ul>
              </div>

              <div className="mb-8">
                <h4 className="font-semibold mb-3 text-accent-blue">Key Features:</h4>
                <ul className="space-y-2">
                  <li className="flex items-center text-sm">
                    <Check className="text-accent-blue mr-3 h-4 w-4" />
                    Multi-format document support
                  </li>
                  <li className="flex items-center text-sm">
                    <Check className="text-accent-blue mr-3 h-4 w-4" />
                    Context-aware AI responses
                  </li>
                  <li className="flex items-center text-sm">
                    <Check className="text-accent-blue mr-3 h-4 w-4" />
                    Real-time chat interface
                  </li>
                  <li className="flex items-center text-sm">
                    <Check className="text-accent-blue mr-3 h-4 w-4" />
                    Document history management
                  </li>
                </ul>
              </div>
              
              <Button 
                className="w-full bg-primary-blue hover:bg-blue-600 transition-colors" 
                onClick={() => handleTryFeature("smart-documentation")}
                data-testid="button-try-smart-doc"
              >
                Try Smart Documentation <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>

            {/* Interview Copilot Card */}
            <div className="bg-card-bg p-8 rounded-2xl border border-border-color hover:border-primary-blue/50 transition-all">
              <div className="w-16 h-16 bg-primary-blue/20 rounded-2xl flex items-center justify-center mb-6">
                <MessageSquare className="text-primary-blue h-8 w-8" />
              </div>
              
              <h3 className="text-2xl font-bold mb-4">Interview Copilot</h3>
              <p className="text-text-secondary mb-6">
                Generate intelligent interview questions from resumes and documents. Prepare better interviews with AI-powered question generation tailored to each candidate.
              </p>
              
              <div className="mb-8">
                <h4 className="font-semibold mb-4 text-primary-blue">How it works:</h4>
                <ul className="space-y-3">
                  <li className="flex items-start">
                    <div className="w-6 h-6 bg-primary-blue rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                      <span className="text-white text-xs font-bold">1</span>
                    </div>
                    <span className="text-sm text-text-secondary">Upload candidate resume or job description</span>
                  </li>
                  <li className="flex items-start">
                    <div className="w-6 h-6 bg-primary-blue rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                      <span className="text-white text-xs font-bold">2</span>
                    </div>
                    <span className="text-sm text-text-secondary">AI analyzes skills, experience, and background</span>
                  </li>
                  <li className="flex items-start">
                    <div className="w-6 h-6 bg-primary-blue rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                      <span className="text-white text-xs font-bold">3</span>
                    </div>
                    <span className="text-sm text-text-secondary">Get customized interview questions with suggested answers</span>
                  </li>
                </ul>
              </div>

              <div className="mb-8">
                <h4 className="font-semibold mb-3 text-accent-blue">Key Features:</h4>
                <ul className="space-y-2">
                  <li className="flex items-center text-sm">
                    <Check className="text-accent-blue mr-3 h-4 w-4" />
                    Auto-generate relevant questions
                  </li>
                  <li className="flex items-center text-sm">
                    <Check className="text-accent-blue mr-3 h-4 w-4" />
                    Customizable difficulty levels
                  </li>
                  <li className="flex items-center text-sm">
                    <Check className="text-accent-blue mr-3 h-4 w-4" />
                    Technical & behavioral questions
                  </li>
                  <li className="flex items-center text-sm">
                    <Check className="text-accent-blue mr-3 h-4 w-4" />
                    Interview flow optimization
                  </li>
                </ul>
              </div>
              
              <Button 
                className="w-full bg-primary-blue hover:bg-blue-600 transition-colors" 
                onClick={() => handleTryFeature("interview-copilot")}
                data-testid="button-try-interview-copilot"
              >
                Try Interview Copilot <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>

          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section id="demo" className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6">See How It Works</h2>
          <p className="text-xl text-text-secondary mb-12">
            Watch our step-by-step demo to understand how SmartDocQ transforms your document workflow
          </p>
          
          <div className="bg-card-bg rounded-2xl p-8 border border-border-color">
            <div className="aspect-video bg-primary-bg rounded-xl flex items-center justify-center mb-6">
              <div className="text-center">
                <div className="w-20 h-20 bg-primary-blue/20 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Play className="text-primary-blue h-10 w-10 ml-1" />
                </div>
                <h3 className="text-xl font-semibold mb-2">Demo Video</h3>
                <p className="text-text-secondary">Coming Soon - Interactive Demo</p>
              </div>
            </div>
            
            <div className="grid md:grid-cols-3 gap-6 text-left">
              <div className="space-y-2">
                <div className="w-8 h-8 bg-primary-blue rounded-full flex items-center justify-center text-sm font-bold">1</div>
                <h4 className="font-semibold">Upload Document</h4>
                <p className="text-sm text-text-secondary">Simply drag and drop your PDF, DOCX, or TXT file</p>
              </div>
              <div className="space-y-2">
                <div className="w-8 h-8 bg-primary-blue rounded-full flex items-center justify-center text-sm font-bold">2</div>
                <h4 className="font-semibold">Ask Questions</h4>
                <p className="text-sm text-text-secondary">Type your questions in natural language</p>
              </div>
              <div className="space-y-2">
                <div className="w-8 h-8 bg-primary-blue rounded-full flex items-center justify-center text-sm font-bold">3</div>
                <h4 className="font-semibold">Get Answers</h4>
                <p className="text-sm text-text-secondary">Receive instant, contextual responses from AI</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Customer Reviews - Scrolling Carousel */}
      <section id="reviews" className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">What Our Users Say</h2>
            <p className="text-xl text-text-secondary">Trusted by professionals worldwide</p>
          </div>

          <div className="relative">
            {/* Navigation Arrows */}
            <button
              onClick={prevReview}
              className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-4 z-10 w-12 h-12 bg-card-bg border border-border-color rounded-full flex items-center justify-center hover:border-primary-blue transition-colors"
              data-testid="button-prev-review"
            >
              <ChevronLeft className="text-primary-blue h-5 w-5" />
            </button>
            
            <button
              onClick={nextReview}
              className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-4 z-10 w-12 h-12 bg-card-bg border border-border-color rounded-full flex items-center justify-center hover:border-primary-blue transition-colors"
              data-testid="button-next-review"
            >
              <ChevronRight className="text-primary-blue h-5 w-5" />
            </button>

            {/* Review Card */}
            <div className="bg-card-bg p-8 rounded-2xl border border-border-color mx-8">
              <div className="text-center max-w-2xl mx-auto">
                <div className="flex justify-center mb-6">
                  {[...Array(reviews[currentReview].rating)].map((_, i) => (
                    <Star key={i} className="text-yellow-400 h-5 w-5 fill-current" />
                  ))}
                </div>
                
                <blockquote className="text-lg text-text-secondary mb-8 italic">
                  "{reviews[currentReview].text}"
                </blockquote>
                
                <div className="flex items-center justify-center">
                  <div className="w-12 h-12 bg-primary-blue rounded-full flex items-center justify-center mr-4">
                    <span className="text-white font-semibold">{reviews[currentReview].avatar}</span>
                  </div>
                  <div className="text-left">
                    <p className="font-semibold text-lg">{reviews[currentReview].name}</p>
                    <p className="text-text-secondary">{reviews[currentReview].role}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Review Indicators */}
            <div className="flex justify-center mt-8 space-x-2">
              {reviews.map((_, index) => (
                <button
                  key={index}
                  onClick={() => setCurrentReview(index)}
                  className={`w-3 h-3 rounded-full transition-colors ${
                    index === currentReview ? "bg-primary-blue" : "bg-border-color"
                  }`}
                  data-testid={`indicator-review-${index}`}
                />
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Contact & Info Section */}
      <section id="contact" className="py-20 px-4 sm:px-6 lg:px-8 bg-card-bg/20">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Get in Touch</h2>
            <p className="text-xl text-text-secondary">Ready to transform your document workflow?</p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-primary-blue/20 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <Mail className="text-primary-blue h-8 w-8" />
              </div>
              <h3 className="font-semibold mb-2">Email Us</h3>
              <p className="text-text-secondary text-sm mb-3">Get support or ask questions</p>
              <a href="mailto:support@smartdocq.com" className="text-primary-blue hover:underline">
                support@smartdocq.com
              </a>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 bg-primary-blue/20 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <Phone className="text-primary-blue h-8 w-8" />
              </div>
              <h3 className="font-semibold mb-2">Call Us</h3>
              <p className="text-text-secondary text-sm mb-3">Mon-Fri, 9AM-6PM EST</p>
              <a href="tel:+1-555-SMARTDQ" className="text-primary-blue hover:underline">
                +1 (555) SMART-DQ
              </a>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 bg-primary-blue/20 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <MapPin className="text-primary-blue h-8 w-8" />
              </div>
              <h3 className="font-semibold mb-2">Visit Us</h3>
              <p className="text-text-secondary text-sm mb-3">Our headquarters</p>
              <address className="text-primary-blue not-italic text-sm">
                123 Innovation Drive<br />
                San Francisco, CA 94105
              </address>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 bg-primary-blue/20 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <MessageSquare className="text-primary-blue h-8 w-8" />
              </div>
              <h3 className="font-semibold mb-2">Social Media</h3>
              <p className="text-text-secondary text-sm mb-3">Follow us for updates</p>
              <div className="flex justify-center space-x-3">
                <a href="#" className="text-primary-blue hover:text-blue-400">Twitter</a>
                <a href="#" className="text-primary-blue hover:text-blue-400">LinkedIn</a>
                <a href="#" className="text-primary-blue hover:text-blue-400">GitHub</a>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
