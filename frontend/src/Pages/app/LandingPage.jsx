import React, { useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import "./AppPages.css";

const LandingPage = () => {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (isAuthenticated) {
      navigate("/app/dashboard", { replace: true });
    }
  }, [isAuthenticated, navigate]);

  return (
    <div className="app-shell app-landing container py-5">
      <div className="app-card p-5 mx-auto app-form-wrap text-center">
        <h1 className="app-title mb-2">Begin Your Practice</h1>
        <p className="app-subtitle mb-4">Track and improve your yoga journey</p>
        <div className="d-flex justify-content-center gap-3">
          <Link to="/app/login" className="btn btn-dark">
            Login
          </Link>
          <Link to="/app/signup" className="btn btn-outline-dark">
            Create Account
          </Link>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
