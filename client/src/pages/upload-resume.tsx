import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Upload as UploadIcon } from "lucide-react";
import { Link } from "wouter";
import Navigation from "@/components/navigation";
import FileUpload from "@/components/file-upload";

export default function UploadResume() {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileSelect = (file: File) => {
    setIsUploading(true);
    
    // Simulate upload process
    setTimeout(() => {
      setUploadedFile(file);
      setIsUploading(false);
      
      // Redirect to Interview Copilot chat after successful upload
      setTimeout(() => {
        window.location.href = "/interview-copilot";
      }, 1000);
    }, 2000);
  };

  return (
    <>
      <Navigation 
        onAuthModal={() => {}} 
        isAuthenticated={true} 
        userEmail="user@example.com" 
      />
      <div className="min-h-screen bg-primary-bg pt-16">
        <div className="max-w-4xl mx-auto px-4 py-16">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold mb-4">
              Interview <span className="text-primary-blue">Copilot</span>
            </h1>
            <p className="text-xl text-text-secondary">
              Upload your resume and get AI-generated interview questions and answers
            </p>
          </div>

          {/* Upload Card */}
          <div className="max-w-2xl mx-auto">
            <div className="bg-card-bg rounded-2xl border border-border-color p-8">
              <div className="text-center mb-8">
                <div className="w-16 h-16 bg-primary-blue rounded-full flex items-center justify-center mx-auto mb-4">
                  <UploadIcon className="text-white h-8 w-8" />
                </div>
                <h2 className="text-2xl font-semibold mb-2">Upload Your Resume</h2>
                <p className="text-text-secondary">
                  Upload your resume or any relevant documents to generate personalized interview questions
                </p>
              </div>

              {!uploadedFile && !isUploading && (
                <FileUpload
                  onFileSelect={handleFileSelect}
                  acceptedTypes=".pdf,.docx,.txt,.doc"
                  maxSize={10}
                  className="min-h-[200px]"
                />
              )}

              {isUploading && (
                <div className="text-center py-12">
                  <div className="w-12 h-12 border-4 border-primary-blue border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                  <h3 className="text-lg font-semibold mb-2">Uploading your resume...</h3>
                  <p className="text-text-secondary">Please wait while we analyze your background</p>
                </div>
              )}

              {uploadedFile && !isUploading && (
                <div className="text-center py-12">
                  <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold mb-2">Upload Successful!</h3>
                  <p className="text-text-secondary mb-4">
                    {uploadedFile.name} has been uploaded successfully. Starting interview preparation...
                  </p>
                  <Link href="/interview-copilot">
                    <Button className="bg-primary-blue hover:bg-blue-600">
                      Start Interview Prep
                    </Button>
                  </Link>
                </div>
              )}

              {!uploadedFile && !isUploading && (
                <div className="mt-8">
                  <Button 
                    className="w-full bg-primary-blue hover:bg-blue-600 text-lg py-6"
                    onClick={() => (document.querySelector('input[type="file"]') as HTMLInputElement)?.click()}
                    data-testid="button-choose-file"
                  >
                    <UploadIcon className="h-5 w-5 mr-2" />
                    Choose File
                  </Button>
                </div>
              )}

              <div className="mt-6 text-center text-sm text-text-secondary">
                <p>Supported formats: PDF, DOCX, TXT, DOC</p>
                <p>Maximum file size: 10MB</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}