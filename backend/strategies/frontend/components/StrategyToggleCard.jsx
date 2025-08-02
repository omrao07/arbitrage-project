import React from "react";

const StrategyToggleCard = ({ strategy, onToggle }) => {
  const {
    name,
    category,
    sharpe_ratio,
    active,
    description,
    performance,
  } = strategy;

  const handleToggle = () => {
    onToggle(strategy.id);
  };

  return (
    <div className="bg-white p-4 rounded-xl shadow-md hover:shadow-lg transition duration-200 border">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="text-lg font-semibold">{name}</h3>
          <p className="text-xs text-gray-400 mb-2 capitalize">{category}</p>
          <p className="text-sm text-gray-600 line-clamp-3">{description}</p>
        </div>
        <div className="flex flex-col items-end">
          <span
            className={`text-sm font-medium px-2 py-1 rounded-full ${
              active ? "bg-green-100 text-green-700" : "bg-red-100 text-red-600"
            }`}
          >
            {active ? "Active" : "Inactive"}
          </span>
          <button
            onClick={handleToggle}
            className={`mt-2 text-sm px-3 py-1 rounded-md ${
              active
                ? "bg-red-500 text-white hover:bg-red-600"
                : "bg-green-500 text-white hover:bg-green-600"
            }`}
          >
            {active ? "Disable" : "Enable"}
          </button>
        </div>
      </div>
      <div className="mt-4 text-xs text-gray-500 flex justify-between">
        <div>Sharpe: {sharpe_ratio?.toFixed(2) ?? "N/A"}</div>
        <div>Return: {(performance * 100).toFixed(2)}%</div>
      </div>
    </div>
  );
};

export default StrategyToggleCard;