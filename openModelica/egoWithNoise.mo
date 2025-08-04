model EgoWithNoise
  // input
  Modelica.Blocks.Interfaces.RealInput accel annotation(
    Placement(visible = true, transformation(origin = {-90, 10}, extent = {{-20, -20}, {20, 20}}, rotation = 0), iconTransformation(origin = {-90, 10}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));
  
  // output
  Modelica.Blocks.Interfaces.RealOutput x annotation(
    Placement(transformation(origin = {100, 84}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {106, 62}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput speed annotation(
    Placement(transformation(origin = {100, -54}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {106, 10}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput accel_out annotation(
    Placement(transformation(origin = {100, 12}, extent = {{-18, -18}, {18, 18}}), iconTransformation(origin = {106, 2}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput noisyX annotation(
    Placement(transformation(origin = {100, 44}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {106, 62}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput noisyAccel_out annotation(
    Placement(transformation(origin = {100, -24}, extent = {{-18, -18}, {18, 18}}), iconTransformation(origin = {106, 2}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput noisySpeed annotation(
    Placement(transformation(origin = {100, -84}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {106, 10}, extent = {{-10, -10}, {10, 10}})));
    
  // Noise generators (one for each input)
  Modelica.Blocks.Noise.NormalNoise normalNoise1(mu = 0, sigma = 0.5, samplePeriod = 0.01, useGlobalSeed = true);
  Modelica.Blocks.Noise.NormalNoise normalNoise2(mu = 0, sigma = 0.3, samplePeriod = 0.01, useGlobalSeed = true);
  Modelica.Blocks.Noise.NormalNoise normalNoise3(mu = 0, sigma = 0.15, samplePeriod = 0.01, useGlobalSeed = true);  
    
  parameter Real x0 = 5;
  parameter Real v0 = 8.33;
  parameter Real v_max = 13.9;  // Velocità massima consentita
  parameter Real v_min = 0; // Velocità minima consentita
  
  Real accel_internal;
  inner Modelica.Blocks.Noise.GlobalSeed globalSeed(fixedSeed = 1234)  annotation(
    Placement(transformation(origin = {-40, 76}, extent = {{-10, -10}, {10, 10}})));
  
initial equation
  x = x0;
  speed = v0;
equation
  der(x) = speed;
// Speed limit logic: if speed exceeds limits, set acceleration to zero
  accel_internal = if (speed > v_max or speed < v_min) then 0 else accel;
  der(speed) = accel_internal;
// Use the internal acceleration
  accel_out = accel_internal;
// Output the actual acceleration
  noisyX = x + normalNoise1.y;
  noisySpeed = speed + normalNoise2.y;
  noisyAccel_out = accel_out + normalNoise3.y;
  annotation(
    uses(Modelica(version = "4.0.0")),
  Diagram);
end EgoWithNoise;