
export interface Turn {
  id: string;
  interviewer: string;
  aiSuggested: string;
  timestamp: number;
  isManualTrigger?: boolean;
  usedSegments?: string[];
}

export interface AppState {
  isActive: boolean;
  history: Turn[];
  currentInterviewerText: string;
  currentAiText: string;
  error: string | null;
  resume: string;
}
