import React from "react";
import { Link } from "react-router-dom";

import annaUniversityLogo from "../../assets/1200px-Anna_University_Logo.svg.png";
import homepageBackground from "../../../assets/homepage_bg.png";
import commonYogaProtocolLogo from "../../../assets/Common_Yoga_Protocol.jpg";

import "./Home.css";

const Home = () => {
  return (
    <main>
      <section
        className="research-hero"
        style={{ "--home-background": `url(${homepageBackground})` }}
      >
        <div className="research-hero__overlay" />

        <div className="research-hero__content">
          <div className="research-hero__heading-row">
            <img
              className="anna-university-logo"
              src={annaUniversityLogo}
              alt="Anna University"
            />

            <div className="research-hero__title-block">
              <p className="eyebrow">
                DIGITAL HEALTH · INTELLIGENT MOVEMENT
              </p>

              <h1>
                <span>Digital Twin and AI-Powered Yoga Assistant</span>
                <br className="research-hero__desktop-break" />
                <span>
                  System with Real-Time Asana Pose Estimation and Corrective
                  Feedback
                </span>
              </h1>
            </div>

            <div className="common-yoga-protocol-mark">
              <img
                className="common-yoga-protocol-logo"
                src={commonYogaProtocolLogo}
                alt="Common Yoga Protocol"
              />
              <span>Common Yoga Protocol</span>
            </div>
          </div>

          <div className="hero-rule" />

          <p className="research-hero__funding">Funded by CMRG</p>

          <p className="investigator">
            <b>Principal Investigator:</b>
            <span className="investigator__name">Dr. S. Chitrakala</span>
            <span>Professor</span>
            <span>Department of Computer Science and Engineering</span>
            <span>College of Engineering Guindy</span>
            <span>Anna University, Chennai</span>
          </p>

          <div
            className="research-hero__brand"
            aria-label="YogaTwin AI"
          >
            <p>YogaTwin AI</p>
            <span>Next Gen Yoga for Wellness</span>
          </div>

          <div className="homeModuleCards">
            <Link
              to="/login"
              className="homeModuleCard homeModuleCard--data"
            >
              <div className="module-icon">▦</div>

              <div>
                <b>Data Collection</b>
                <p>For lab operators and researchers</p>
              </div>

              <strong>→</strong>
            </Link>

            <Link
              to="/app"
              className="homeModuleCard homeModuleCard--practice"
            >
              <div className="module-icon">◌</div>

              <div>
                <b>Yoga Practice</b>
                <p>For individual users and learners</p>
              </div>

              <strong>→</strong>
            </Link>
          </div>
        </div>
      </section>
    </main>
  );
};

export default Home;