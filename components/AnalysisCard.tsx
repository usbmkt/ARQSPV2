
import React from 'react';

interface AnalysisCardProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

const AnalysisCard: React.FC<AnalysisCardProps> = ({ title, icon, children, className }) => {
  return (
    <div className={`bg-gray-800 border border-gray-700 rounded-lg p-4 md:p-6 shadow-lg ${className}`}>
      <div className="flex items-center mb-4">
        <div className="text-gray-400 mr-3">{icon}</div>
        <h3 className="font-teko text-2xl md:text-3xl font-medium tracking-wide uppercase">{title}</h3>
      </div>
      <div className="text-gray-300 space-y-2 text-sm md:text-base">
        {children}
      </div>
    </div>
  );
};

export default AnalysisCard;
