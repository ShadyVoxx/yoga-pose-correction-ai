import React from "react";
import { Navigate } from "react-router-dom";
import { useSession } from "../context/SessionContext";

const hasOperatorInfo = (operatorInfo) =>
  Boolean(operatorInfo?.operatorName?.trim() && operatorInfo?.institutionName?.trim());

const hasMetadata = (metadata) =>
  Boolean(
    metadata?.age &&
      metadata?.gender &&
      metadata?.height &&
      metadata?.weight &&
      metadata?.experience
  );

const RequireSession = ({ step, children }) => {
  const { operatorInfo, metadata, consentGiven } = useSession();

  if (step === "metadata" && !hasOperatorInfo(operatorInfo)) {
    return <Navigate to="/login" replace />;
  }

  if (step === "consent" && !hasMetadata(metadata)) {
    return <Navigate to="/metadata" replace />;
  }

  if (step === "hardware" && !consentGiven) {
    return <Navigate to="/consent" replace />;
  }

  return children;
};

export default RequireSession;
