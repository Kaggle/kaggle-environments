export interface ReplayStep {
  observation: Record<string, any>;
  action: Record<string, any> | null;
  reward: Record<string, number> | number | null;
  info: Record<string, any>;
  status: string;
}

export interface ReplayData {
  name: string;
  version: string;
  steps: ReplayStep[][];
  configuration: Record<string, any>;
  info?: Record<string, any>;
}
