import { Switch, Route, useLocation } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useState, useEffect } from "react";

import NotFound from "@/pages/not-found";
import Landing from "@/pages/landing";
import Features from "@/pages/features";
import SmartDocumentation from "@/pages/smart-documentation";
import InterviewCopilot from "@/pages/interview-copilot";
import InterviewReview from "@/pages/interview-review";
import Upload from "@/pages/upload";
import UploadResume from "@/pages/upload-resume";
import Login from "@/pages/login";
import Signup from "@/pages/signup";

import Navigation from "@/components/navigation";
import Footer from "@/components/footer";
import AuthModal from "@/components/auth-modal";

// Router Component
function Router({
  isAuthenticated,
  userEmail,
  onAuthModal,
  onLogout,
}: {
  isAuthenticated: boolean;
  userEmail: string;
  onAuthModal: (type: "login" | "signup") => void;
  onLogout: () => void;
}) {
  return (
    <Switch>
      <Route path="/">
        <Landing isAuthenticated={isAuthenticated} onAuthModal={onAuthModal} />
      </Route>
      <Route path="/features" component={Features} />
      <Route path="/smart-documentation">
        <SmartDocumentation 
          isAuthenticated={isAuthenticated}
          userEmail={userEmail}
          onLogout={onLogout}
        />
      </Route>
      <Route path="/interview-copilot">
        <InterviewCopilot 
          isAuthenticated={isAuthenticated}
          userEmail={userEmail}
          onLogout={onLogout}
        />
      </Route>
      <Route path="/interview-review" component={InterviewReview} />
      <Route path="/upload" component={Upload} />
      <Route path="/upload-resume" component={UploadResume} />
      <Route path="/login" component={Login} />
      <Route path="/signup" component={Signup} />
      <Route component={NotFound} />
    </Switch>
  );
}

// Main App Component
function App() {
  const [authModal, setAuthModal] = useState<"login" | "signup" | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userEmail, setUserEmail] = useState<string>("");

  const [location] = useLocation();

  // ✅ On successful login/signup
  const handleAuthSuccess = (email: string) => {
    setIsAuthenticated(true);
    setUserEmail(email);
    localStorage.setItem("token", "dummy_token"); // save token if needed
    localStorage.setItem("userEmail", email);
    setAuthModal(null);
  };

  // ✅ Logout function
  const handleLogout = () => {
    setIsAuthenticated(false);
    setUserEmail("");
    localStorage.removeItem("token");
    localStorage.removeItem("userEmail");
  };

  // Load auth state from localStorage on mount
  useEffect(() => {
    const token = localStorage.getItem("token");
    const email = localStorage.getItem("userEmail");
    if (token && email) {
      setIsAuthenticated(true);
      setUserEmail(email);
    }
  }, []);

  const fullscreenRoutes = [
    "/smart-documentation",
    "/interview-copilot",
    "/upload",
    "/upload-resume",
  ];
  const isFullscreenRoute = fullscreenRoutes.includes(location);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <div className="min-h-screen bg-primary-bg text-white">
          {!isFullscreenRoute && (
            <Navigation
              onAuthModal={setAuthModal}
              onLogout={handleLogout}      // ✅ pass logout here
              isAuthenticated={isAuthenticated}
              userEmail={userEmail}
            />
          )}

          <main className={!isFullscreenRoute ? "pt-16" : ""}>
            <Router
              isAuthenticated={isAuthenticated}
              userEmail={userEmail}
              onAuthModal={setAuthModal}
              onLogout={handleLogout}
            />
          </main>

          {!isFullscreenRoute && <Footer />}

          <AuthModal
            type={authModal}
            onClose={() => setAuthModal(null)}
            onSwitch={(type) => setAuthModal(type)}
            onAuthSuccess={handleAuthSuccess}
          />

          <Toaster />
        </div>
      </TooltipProvider>
    </QueryClientProvider>
  );
}
export default App;