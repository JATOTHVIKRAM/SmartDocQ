import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { zodResolver } from "@hookform/resolvers/zod";
import { ArrowRight, Lock, Mail, User } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";

const loginSchema = z.object({
  email: z.string().email("Invalid email address"),
  password: z.string().min(8, "Password must be at least 8 characters"),
});

const signupSchema = z.object({
  fullName: z.string().min(2, "Full name must be at least 2 characters"),
  email: z.string().email("Invalid email address"),
  password: z.string().min(8, "Password must be at least 8 characters"),
  confirmPassword: z.string().min(8, "Password confirmation is required"),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

interface AuthModalProps {
  type: "login" | "signup" | null;
  onClose: () => void;
  onSwitch: (type: "login" | "signup") => void;
  onAuthSuccess?: (email: string,token?:string) => void;
}

export default function AuthModal({ type, onClose, onSwitch, onAuthSuccess }: AuthModalProps) {
  const [isLoading, setIsLoading] = useState(false);

  const loginForm = useForm({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  });

  const signupForm = useForm({
    resolver: zodResolver(signupSchema),
    defaultValues: {
      fullName: "",
      email: "",
      password: "",
      confirmPassword: "",
    },
  });

  const onLoginSubmit = async (data: z.infer<typeof loginSchema>) => {
    setIsLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const result = await res.json();
      if (res.ok) {
        // ✅ Save token and email to localStorage
        localStorage.setItem("token", result.token || "dummy_token"); // use API token if available
        localStorage.setItem("userEmail", result.user.email);

        // ✅ Update App state
        onAuthSuccess?.(result.user.email);

        console.log("Login successful:", result.user.email);
        onClose();
      } else {
        throw new Error(result.error || "Login failed");
      }
    } catch (error) {
      alert(error);
      console.error("Login error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const onSignupSubmit = async (data: z.infer<typeof signupSchema>) => {
    setIsLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          full_name: data.fullName,
          email: data.email,
          password: data.password,
          confirm_password: data.confirmPassword,
        }),
      });
      const result = await res.json();
      if (res.ok) {
        // ✅ Save token and email to localStorage
        localStorage.setItem("token", result.token || "dummy_token");
        localStorage.setItem("userEmail", result.email);

        // ✅ Update App state
        onAuthSuccess?.(result.email);

        console.log("Signup successful:", result.email);
        onClose();
      } else {
        throw new Error(result.error || "Signup failed");
      }
    } catch (error) {
      alert(error);
      console.error("Signup error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  if (!type) return null;

  return (
    <Dialog open={!!type} onOpenChange={() => onClose()}>
      <DialogContent className="bg-card-bg border-border-color max-w-md">
        <DialogHeader>
          <DialogTitle className="text-center text-2xl font-bold mb-2">
            {type === "login" ? "Sign In" : "Create Account"}
          </DialogTitle>
          {type === "login" && (
            <p className="text-text-secondary text-center">Welcome back</p>
          )}
        </DialogHeader>

        {type === "login" ? (
          <form onSubmit={loginForm.handleSubmit(onLoginSubmit)} className="space-y-6">
            <div>
              <Label className="block text-sm font-medium mb-2">Email Address</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 text-text-secondary h-4 w-4" />
                <Input
                  {...loginForm.register("email")}
                  type="email"
                  placeholder="Enter your email"
                  className="w-full bg-primary-bg border-border-color pl-10 focus:border-primary-blue"
                  data-testid="input-email"
                />
              </div>
              {loginForm.formState.errors.email && (
                <p className="text-destructive text-sm mt-1">{loginForm.formState.errors.email.message}</p>
              )}
            </div>
            <div>
              <Label className="block text-sm font-medium mb-2">Password</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-text-secondary h-4 w-4" />
                <Input
                  {...loginForm.register("password")}
                  type="password"
                  placeholder="Enter your password"
                  className="w-full bg-primary-bg border-border-color pl-10 focus:border-primary-blue"
                  data-testid="input-password"
                />
              </div>
              {loginForm.formState.errors.password && (
                <p className="text-destructive text-sm mt-1">{loginForm.formState.errors.password.message}</p>
              )}
            </div>
            <Button
              type="submit"
              disabled={isLoading}
              className="w-full bg-primary-blue hover:bg-blue-600 transition-colors"
              data-testid="button-submit-login"
            >
              {isLoading ? "Signing In..." : "Sign In"} <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </form>
        ) : (
          <form onSubmit={signupForm.handleSubmit(onSignupSubmit)} className="space-y-6">
            <div>
              <Label className="block text-sm font-medium mb-2">Full Name</Label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 transform -translate-y-1/2 text-text-secondary h-4 w-4" />
                <Input
                  {...signupForm.register("fullName")}
                  type="text"
                  placeholder="Enter your full name"
                  className="w-full bg-primary-bg border-border-color pl-10 focus:border-primary-blue"
                  data-testid="input-fullname"
                />
              </div>
              {signupForm.formState.errors.fullName && (
                <p className="text-destructive text-sm mt-1">{signupForm.formState.errors.fullName.message}</p>
              )}
            </div>
            <div>
              <Label className="block text-sm font-medium mb-2">Email Address</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 text-text-secondary h-4 w-4" />
                <Input
                  {...signupForm.register("email")}
                  type="email"
                  placeholder="Enter your email"
                  className="w-full bg-primary-bg border-border-color pl-10 focus:border-primary-blue"
                  data-testid="input-email"
                />
              </div>
              {signupForm.formState.errors.email && (
                <p className="text-destructive text-sm mt-1">{signupForm.formState.errors.email.message}</p>
              )}
            </div>
            <div>
              <Label className="block text-sm font-medium mb-2">Password</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-text-secondary h-4 w-4" />
                <Input
                  {...signupForm.register("password")}
                  type="password"
                  placeholder="Create a password"
                  className="w-full bg-primary-bg border-border-color pl-10 focus:border-primary-blue"
                  data-testid="input-password"
                />
              </div>
              {signupForm.formState.errors.password && (
                <p className="text-destructive text-sm mt-1">{signupForm.formState.errors.password.message}</p>
              )}
            </div>
            <div>
              <Label className="block text-sm font-medium mb-2">Confirm Password</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-text-secondary h-4 w-4" />
                <Input
                  {...signupForm.register("confirmPassword")}
                  type="password"
                  placeholder="Confirm your password"
                  className="w-full bg-primary-bg border-border-color pl-10 focus:border-primary-blue"
                  data-testid="input-confirm-password"
                />
              </div>
              {signupForm.formState.errors.confirmPassword && (
                <p className="text-destructive text-sm mt-1">{signupForm.formState.errors.confirmPassword.message}</p>
              )}
            </div>
            <Button
              type="submit"
              disabled={isLoading}
              className="w-full bg-primary-blue hover:bg-blue-600 transition-colors"
              data-testid="button-submit-signup"
            >
              {isLoading ? "Creating Account..." : "Create Account"} <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </form>
        )}

        <div className="text-center mt-6">
          <p className="text-text-secondary">
            {type === "login" ? "Don't have an account? " : "Already have an account? "}
            <button
              onClick={() => onSwitch(type === "login" ? "signup" : "login")}
              className="text-primary-blue hover:underline"
              data-testid="button-switch-auth"
            >
              {type === "login" ? "Sign up" : "Sign in"}
            </button>
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}