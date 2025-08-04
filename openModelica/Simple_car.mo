model Simple_car
  Modelica.Blocks.Interfaces.RealOutput x annotation(
    Placement(transformation(origin = {100, 80}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {106, 62}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput speed annotation(
    Placement(transformation(origin = {100, -32}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {106, 10}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealInput accel annotation(
    Placement(visible = true, transformation(origin = {-90, 10}, extent = {{-20, -20}, {20, 20}}, rotation = 0), iconTransformation(origin = {-90, 10}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));
  parameter Real x0 = 3;
  parameter Real v0 = -0.1;
  Modelica.Blocks.Interfaces.RealOutput accel_out annotation(
    Placement(transformation(origin = {100, 16}, extent = {{-18, -18}, {18, 18}}), iconTransformation(origin = {106, 2}, extent = {{-10, -10}, {10, 10}})));
initial equation
  x = x0;
  speed = v0;
equation
  der(x) = speed;
  der(speed) = accel;
  accel_out = accel;
  annotation(
    uses(Modelica(version = "4.0.0")));
end Simple_car;