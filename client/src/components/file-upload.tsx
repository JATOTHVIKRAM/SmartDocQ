import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Upload, File, X } from "lucide-react";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  acceptedTypes?: string;
  maxSize?: number; // in MB
  className?: string;
}

export default function FileUpload({ 
  onFileSelect, 
  acceptedTypes = ".pdf,.docx,.txt", 
  maxSize = 10,
  className = ""
}: FileUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    setError("");
    
    // Validate file size
    if (file.size > maxSize * 1024 * 1024) {
      setError(`File size must be less than ${maxSize}MB`);
      return;
    }

    // Validate file type
    const fileExtension = "." + file.name.split(".").pop()?.toLowerCase();
    if (!acceptedTypes.includes(fileExtension)) {
      setError("File type not supported");
      return;
    }

    setSelectedFile(file);
    onFileSelect(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const removeFile = () => {
    setSelectedFile(null);
    setError("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className={className}>
      <div
        className={`border-2 border-dashed rounded-2xl p-12 text-center transition-colors ${
          isDragOver
            ? "border-primary-blue bg-primary-blue/10"
            : "border-border-color hover:border-primary-blue/50"
        } ${selectedFile ? "bg-card-bg/50" : "bg-card-bg"}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        data-testid="file-upload-area"
      >
        {selectedFile ? (
          <div className="space-y-4">
            <div className="flex items-center justify-center space-x-3">
              <File className="text-primary-blue h-8 w-8" />
              <div className="text-left">
                <p className="font-medium" data-testid="text-filename">{selectedFile.name}</p>
                <p className="text-sm text-text-secondary">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={removeFile}
                className="text-text-secondary hover:text-white"
                data-testid="button-remove-file"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
        ) : (
          <div>
            <div className="w-16 h-16 bg-primary-blue/20 rounded-full flex items-center justify-center mx-auto mb-6">
              <Upload className="text-primary-blue h-8 w-8" />
            </div>
            <h3 className="text-2xl font-semibold mb-4">Upload Your File</h3>
            <p className="text-text-secondary mb-8">
              Drag and drop your file here, or click to browse
            </p>
            <Button
              onClick={() => fileInputRef.current?.click()}
              className="bg-primary-blue hover:bg-blue-600 transition-colors"
              data-testid="button-choose-file"
            >
              <Upload className="mr-2 h-4 w-4" />
              Choose File
            </Button>
          </div>
        )}
        
        <input
          ref={fileInputRef}
          type="file"
          accept={acceptedTypes}
          onChange={handleFileInputChange}
          className="hidden"
          data-testid="input-file"
        />
      </div>
      
      {error && (
        <p className="text-destructive text-sm mt-4 text-center" data-testid="text-error">
          {error}
        </p>
      )}
      
      <p className="text-sm text-text-secondary mt-4 text-center">
        Supports {acceptedTypes.replace(/\./g, "").toUpperCase()} files up to {maxSize}MB
      </p>
    </div>
  );
}
