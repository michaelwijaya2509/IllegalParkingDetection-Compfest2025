'use client';
import { useState } from "react";
import { DataContext } from "../context/Context";

export const ContextProvider = ({ children }: { children: React.ReactNode }) => {
  const [selectedCamId, setSelectedCamId] = useState<string | null>(null);

  return (
    <DataContext.Provider value={{ selectedCamId, setSelectedCamId }}>
      {children}
    </DataContext.Provider>
  );
};