import React from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";

const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, hydrating } = useAuth();

  if (hydrating) {
    return <div className="container py-4">Loading...</div>;
  }

  if (!isAuthenticated) {
    return <Navigate to="/app/login" replace />;
  }

  return children;
};

export default ProtectedRoute;
