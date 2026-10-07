import React from "react";
import { useNavigate } from "react-router-dom";
import img7 from "../../assets/yoga9.jpg";

import "./Yog1.css";

const Yog6 = () => {
  const navigate = useNavigate();

  function startYoga() {
    navigate("/practice");
  }

  return (
    <div className="yogContainer p-3  ">
      <h1 className="text-center my-3">Downward Dog Pose</h1>
      <div className="yogContainerData d-flex gap-5 ">
        <div className="yogInfo gap-5 ">
          <ul className="">
            <li>Great alternative to Downward Dog pose if you have sensitive wrists.
            </li>
            <li>Strengthens the shoulders, arms, upper body and legs.
            </li>
            <li>Activates the arches of your feet.
            </li>
            <li>Dolphin pose gives you the strength and actions needed for Headstand and Forearm balance.</li>
            <li>You should be able to hold the pose for 20 breaths before working on those poses.</li>

          </ul>
        </div>
        <div className="yogImg m-auto d-flex justify-content-center align-items-center ">
          <img
            className=""
            src={img7}
            alt=""
          />
        </div>
      </div>
      <div className="mt-5 yogStartPose my-2 ">
      <button  onClick={startYoga} className="btn btn-primary w-20 my-6">Start Pose</button>
      </div>
    </div>
  );
};

export default Yog6;
