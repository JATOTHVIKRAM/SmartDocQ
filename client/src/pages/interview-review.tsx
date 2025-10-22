import { useEffect, useState } from "react";
import Navigation from "@/components/navigation";

export default function InterviewReview() {
  const [review, setReview] = useState<null | {
    session_id?: string;
    avg_score?: number | null;
    feedback?: Array<{ question: string; user_answer: string; score: number; feedback: string; correct_answer?: string }>;
  }>(null);

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem("interview_review");
      if (raw) {
        setReview(JSON.parse(raw));
      }
    } catch (e) {
      console.error("Failed to load review from sessionStorage", e);
    }
  }, []);

  if (!review) {
    return (
      <div className="min-h-screen pt-16">
        <Navigation onAuthModal={() => {}} isAuthenticated={true} userEmail={localStorage.getItem("userEmail") || "User"} />
        <div className="p-8 text-center text-text-secondary">No review found. Please complete an interview first.</div>
      </div>
    );
  }

  const feedback = review.feedback || [];

  return (
    <>
      <Navigation onAuthModal={() => {}} isAuthenticated={true} userEmail={localStorage.getItem("userEmail") || "User"} />
      <div className="pt-16 p-6 max-w-4xl mx-auto">
        <h2 className="text-2xl font-semibold mb-4">Interview Review</h2>
        <div className="mb-4">
          <strong>Session:</strong> {review.session_id}
        </div>
        <div className="mb-6">
          <strong>Average Score:</strong> {review.avg_score ?? "-"}
        </div>

        <div className="space-y-4">
          {feedback.map((fb, i) => (
            <div key={i} className="p-4 bg-card-bg border border-border-color rounded">
              <div className="font-medium">{i + 1}. {fb.question}</div>
              <div className="text-sm text-text-secondary mt-1"><strong>Your answer:</strong> {fb.user_answer}</div>
              {fb.correct_answer && (
                <div className="text-sm text-green-300 mt-1"><strong>Correct answer:</strong> {fb.correct_answer}</div>
              )}
              <div className="text-sm mt-1"><strong>Score:</strong> {fb.score}/10</div>
              <div className="text-sm mt-1 text-text-secondary"><strong>Feedback:</strong> {fb.feedback}</div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
