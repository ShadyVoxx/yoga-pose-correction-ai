import React from "react";
import "./Yog1.css";
import { Link } from "react-router-dom";
import  yoga25 from "../../assets/yoga25.jpg"
import { useNavigate } from "react-router-dom";
const Yog2 = () => {
  const navigate = useNavigate();

  function startYoga() {
    navigate("/practice");
  }

  return (
    <div className="yogContainer p-3  ">
      <h1 className="text-center my-3">T Pose</h1>
      <div className="yogContainerData d-flex gap-5 ">
        <div className="yogInfo gap-5 ">
          <ul className="">
            <li>Improves your flexibility</li>
            <li>Strengthens your legs and abs</li>
            <li> Keeps your digestive tract on track</li>
            <li>Develops better balance</li>
          </ul>
        </div>
        <div className="yogImg m-auto d-flex justify-content-center align-items-center ">
          <img
            className=""
            src={yoga25}
            alt=""
          />
        </div>
      </div>
      {/* <Link to=""> */}
      <div className="yogStartPose my-2 mt-5   ">
        <button onClick={startYoga} className="btn btn-primary w-20 my-6">
          Start Pose
        </button>
      </div>
      {/* </Link> */}
    </div>
  );
};

export default Yog2;

