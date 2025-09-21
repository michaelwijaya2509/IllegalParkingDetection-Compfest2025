import { createContext, Dispatch, SetStateAction } from "react";

interface DataContextType {
  selectedCamId: string | null;
  setSelectedCamId: Dispatch<SetStateAction<string | null>>;
}

export const DataContext = createContext<DataContextType | null>(null);