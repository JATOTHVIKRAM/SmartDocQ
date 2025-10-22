import { useState } from "react";
import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { User, Mail, Lock, ArrowRight } from "lucide-react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";

const signupSchema = z.object({
  fullName: z.string().min(2, "Full name must be at least 2 characters"),
  email: z.string().email("Invalid email address"),
  password: z.string().min(6, "Password must be at least 6 characters"),
  confirmPassword: z.string().min(6, "Password confirmation is required"),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

export default function Signup() {
  const [isLoading, setIsLoading] = useState(false);

  const form = useForm({
    resolver: zodResolver(signupSchema),
    defaultValues: {
      fullName: "",
      email: "",
      password: "",
      confirmPassword: "",
    },
  });

  const onSubmit = async (data: z.infer<typeof signupSchema>) => {
    setIsLoading(true);
    try {
      // TODO: Implement signup API call
      console.log("Signup data:", data);
    } catch (error) {
      console.error("Signup error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-md bg-card-bg border-border-color">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl font-bold mb-2">Create Account</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
            <div>
              <Label className="block text-sm font-medium mb-2">Full Name</Label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 transform -translate-y-1/2 text-text-secondary h-4 w-4" />
                <Input
                  {...form.register("fullName")}
                  type="text"
                  placeholder="Enter your full name"
                  className="w-full bg-primary-bg border-border-color pl-10 focus:border-primary-blue"
                  data-testid="input-fullname"
                />
              </div>
              {form.formState.errors.fullName && (
                <p className="text-destructive text-sm mt-1">{form.formState.errors.fullName.message}</p>
              )}
            </div>
            <div>
              <Label className="block text-sm font-medium mb-2">Email Address</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 text-text-secondary h-4 w-4" />
                <Input
                  {...form.register("email")}
                  type="email"
                  placeholder="Enter your email"
                  className="w-full bg-primary-bg border-border-color pl-10 focus:border-primary-blue"
                  data-testid="input-email"
                />
              </div>
              {form.formState.errors.email && (
                <p className="text-destructive text-sm mt-1">{form.formState.errors.email.message}</p>
              )}
            </div>
            <div>
              <Label className="block text-sm font-medium mb-2">Password</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-text-secondary h-4 w-4" />
                <Input
                  {...form.register("password")}
                  type="password"
                  placeholder="Create a password"
                  className="w-full bg-primary-bg border-border-color pl-10 focus:border-primary-blue"
                  data-testid="input-password"
                />
              </div>
              {form.formState.errors.password && (
                <p className="text-destructive text-sm mt-1">{form.formState.errors.password.message}</p>
              )}
            </div>
            <div>
              <Label className="block text-sm font-medium mb-2">Confirm Password</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-text-secondary h-4 w-4" />
                <Input
                  {...form.register("confirmPassword")}
                  type="password"
                  placeholder="Confirm your password"
                  className="w-full bg-primary-bg border-border-color pl-10 focus:border-primary-blue"
                  data-testid="input-confirm-password"
                />
              </div>
              {form.formState.errors.confirmPassword && (
                <p className="text-destructive text-sm mt-1">{form.formState.errors.confirmPassword.message}</p>
              )}
            </div>
            <Button
              type="submit"
              disabled={isLoading}
              className="w-full bg-primary-blue hover:bg-blue-600 transition-colors"
              data-testid="button-submit"
            >
              {isLoading ? "Creating Account..." : "Create Account"} <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </form>

          <div className="text-center mt-6">
            <p className="text-text-secondary">
              Already have an account?{" "}
              <Link href="/login" className="text-primary-blue hover:underline" data-testid="link-login">
                Sign in
              </Link>
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
