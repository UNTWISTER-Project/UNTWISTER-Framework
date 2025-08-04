model Dummy
  Modelica.Blocks.Interfaces.RealOutput x annotation(
    Placement(visible = true, transformation(origin = {100, 52}, extent = {{-20, -20}, {20, 20}}, rotation = 0), iconTransformation(origin = {106, 62}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput speed annotation(
    Placement(visible = true, transformation(origin = {100, -60}, extent = {{-20, -20}, {20, 20}}, rotation = 0), iconTransformation(origin = {106, 10}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealInput accel annotation(
    Placement(visible = true, transformation(origin = {-90, 10}, extent = {{-20, -20}, {20, 20}}, rotation = 0), iconTransformation(origin = {-90, 10}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));

  parameter Real x0 = 3;
  parameter Real v0 = -0.1;
  
  Modelica.Blocks.Interfaces.RealOutput accel_out annotation(
    Placement(visible = true, transformation(origin = {98, 10}, extent = {{-18, -18}, {18, 18}}, rotation = 0), iconTransformation(origin = {106, 2}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
initial equation
  x = x0;
  speed = v0;
equation
  der(x) = speed;
  der(speed) = accel;
  accel_out = accel;
end Dummy;