import { Link } from "wouter";
import { DollarSign, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";

interface NavigationProps {
  onAuthModal: (type: "login" | "signup") => void;
 // onLogout?:()=>void;
  isAuthenticated?: boolean;
  userEmail?: string;
  onLogout?: () => void;
}

export default function Navigation({
  onAuthModal,
  isAuthenticated = false,
  userEmail,
  onLogout,
}: NavigationProps) {
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const scrollToSection = (sectionId: string) => {
    document.getElementById(sectionId)?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-primary-bg/95 backdrop-blur-sm border-b border-border-color">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center cursor-pointer">
            <div className="w-8 h-8 bg-primary-blue rounded-lg flex items-center justify-center mr-3">
              <DollarSign className="text-white text-lg" />
            </div>
            <span className="text-xl font-semibold">SmartDocQ</span>
          </Link>

          {/* Links */}
          <div className="hidden md:flex items-center space-x-6">
            <button
              onClick={() => scrollToSection("features")}
              className="text-text-secondary hover:text-white transition-colors"
            >
              Features
            </button>
            <button
              onClick={() => scrollToSection("demo")}
              className="text-text-secondary hover:text-white transition-colors"
            >
              Demo
            </button>
            <button
              onClick={() => scrollToSection("reviews")}
              className="text-text-secondary hover:text-white transition-colors"
            >
              Reviews
            </button>
            <button
              onClick={() => scrollToSection("contact")}
              className="text-text-secondary hover:text-white transition-colors"
            >
              Contact
            </button>
          </div>

          {/* Auth Section */}
          <div className="relative flex items-center space-x-4">
            {isAuthenticated ? (
              <div className="relative">
                <button
                  onClick={() => setDropdownOpen((prev) => !prev)}
                  className="flex items-center space-x-2 focus:outline-none"
                >
                  <div className="w-8 h-8 bg-primary-blue rounded-full flex items-center justify-center">
                    <User className="h-4 w-4 text-white" />
                  </div>
                  <span className="text-sm">{userEmail}</span>
                </button>

                {/* Dropdown */}
                {dropdownOpen && (
                  <div className="absolute right-0 mt-2 w-32 bg-card-bg border border-border-color rounded-lg shadow-lg py-2 z-50">
                    <button
                      onClick={() => {
                        setDropdownOpen(false);
                        onLogout?.();
                      }}
                      className="block w-full text-left px-4 py-2 text-sm text-white hover:bg-primary-blue/20"
                    >
                      Logout
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <>
                <Button variant="ghost" onClick={() => onAuthModal("login")}>
                  Login
                </Button>
                <Button onClick={() => onAuthModal("signup")}>Sign Up</Button>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}