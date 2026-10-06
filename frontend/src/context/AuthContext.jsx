import React, { createContext, useContext, useEffect, useMemo, useState } from "react";

const AuthContext = createContext(null);
const API_BASE = "http://127.0.0.1:3001";

export const AuthContextProvider = ({ children }) => {
  const [token, setToken] = useState(() => localStorage.getItem("yoga_app_token") || "");
  const [currentUser, setCurrentUser] = useState(null);
  const [hydrating, setHydrating] = useState(true);

  const logout = () => {
    setToken("");
    setCurrentUser(null);
    localStorage.removeItem("yoga_app_token");
  };

  const login = (nextToken, user) => {
    setToken(nextToken);
    setCurrentUser(user);
    localStorage.setItem("yoga_app_token", nextToken);
  };

  useEffect(() => {
    let cancelled = false;
    if (!token) {
      setHydrating(false);
      return undefined;
    }

    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/auth/me`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) throw new Error("auth failed");
        const data = await res.json();
        if (!cancelled) {
          setCurrentUser(data.user || null);
        }
      } catch {
        if (!cancelled) {
          logout();
        }
      } finally {
        if (!cancelled) setHydrating(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [token]);

  const value = useMemo(
    () => ({
      currentUser,
      token,
      login,
      logout,
      isAuthenticated: Boolean(token && currentUser),
      hydrating,
    }),
    [currentUser, token, hydrating]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within AuthContextProvider");
  return context;
};
