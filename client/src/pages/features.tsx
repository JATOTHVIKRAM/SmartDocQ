import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { FileText, MessageSquare, ArrowRight, Check } from "lucide-react";

export default function Features() {
  return (
    <div className="min-h-screen">
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-2 gap-8">
            
            {/* Smart Documentation Card */}
            <div className="bg-card-bg p-8 rounded-2xl border border-border-color hover:border-primary-blue/50 transition-all">
              <div className="w-16 h-16 bg-primary-blue/20 rounded-2xl flex items-center justify-center mb-6">
                <FileText className="text-primary-blue h-8 w-8" />
              </div>
              
              <h3 className="text-2xl font-bold mb-4">Smart Documentation</h3>
              <p className="text-text-secondary mb-6">
                Upload any document and get intelligent, contextual answers instantly.
              </p>
              
              {/* Features List */}
              <ul className="space-y-3 mb-8">
                <li className="flex items-center text-sm">
                  <Check className="text-accent-blue mr-3 h-4 w-4" />
                  Support for PDF, DOCX, TXT, and more
                </li>
                <li className="flex items-center text-sm">
                  <Check className="text-accent-blue mr-3 h-4 w-4" />
                  AI-powered content understanding
                </li>
                <li className="flex items-center text-sm">
                  <Check className="text-accent-blue mr-3 h-4 w-4" />
                  Real-time Q&A interface
                </li>
              </ul>
              
              <Link href="/smart-documentation">
                <Button className="w-full bg-primary-blue hover:bg-blue-600 transition-colors" data-testid="button-try-smart-doc">
                  Try Smart Documentation <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </div>

            {/* Interview Copilot Card */}
            <div className="bg-card-bg p-8 rounded-2xl border border-border-color hover:border-primary-blue/50 transition-all">
              <div className="w-16 h-16 bg-primary-blue/20 rounded-2xl flex items-center justify-center mb-6">
                <MessageSquare className="text-primary-blue h-8 w-8" />
              </div>
              
              <h3 className="text-2xl font-bold mb-4">Interview Copilot</h3>
              <p className="text-text-secondary mb-6">
                Generate intelligent interview questions from any document or resume.
              </p>
              
              {/* Features List */}
              <ul className="space-y-3 mb-8">
                <li className="flex items-center text-sm">
                  <Check className="text-accent-blue mr-3 h-4 w-4" />
                  Auto-generate relevant questions
                </li>
                <li className="flex items-center text-sm">
                  <Check className="text-accent-blue mr-3 h-4 w-4" />
                  Customizable question difficulty
                </li>
                <li className="flex items-center text-sm">
                  <Check className="text-accent-blue mr-3 h-4 w-4" />
                  Interview flow optimization
                </li>
              </ul>
              
              <Link href="/interview-copilot">
                <Button className="w-full bg-primary-blue hover:bg-blue-600 transition-colors" data-testid="button-try-interview-copilot">
                  Try Interview Copilot <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </div>

          </div>
        </div>
      </section>
    </div>
  );
}
