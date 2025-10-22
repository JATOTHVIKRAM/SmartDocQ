import { Brain } from "lucide-react";

export default function Footer() {
  return (
    <footer className="bg-card-bg border-t border-border-color py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center mb-4 md:mb-0">
            <div className="w-8 h-8 bg-primary-blue rounded-lg flex items-center justify-center mr-3">
              <Brain className="text-white h-5 w-5" />
            </div>
            <span className="text-xl font-semibold">SmartDocQ</span>
          </div>
          <div className="flex space-x-6 text-text-secondary text-sm">
            <a href="#" className="hover:text-accent-blue transition-colors" data-testid="link-privacy">
              Privacy Policy
            </a>
            <a href="#" className="hover:text-accent-blue transition-colors" data-testid="link-terms">
              Terms of Service
            </a>
            <a href="#" className="hover:text-accent-blue transition-colors" data-testid="link-support">
              Support
            </a>
          </div>
        </div>
        <div className="text-center text-text-secondary text-sm mt-8">
          <p>© 2025 SmartDocQ. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}
