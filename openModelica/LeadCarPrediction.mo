model LeadCarPrediction
  Simple_car simple_car(v0 = 20, x0 = 30)  annotation(
    Placement(transformation(origin = {-18, 6}, extent = {{-24, -24}, {24, 24}})));
  Modelica.Blocks.Interfaces.RealOutput x annotation(
    Placement(visible = true, transformation(origin = {120, 72}, extent = {{-20, -20}, {20, 20}}, rotation = 0), iconTransformation(origin = {108, 66}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput speed annotation(
    Placement(visible = true, transformation(origin = {120, 8}, extent = {{-20, -20}, {20, 20}}, rotation = 0), iconTransformation(origin = {106, -74}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput outputAccel annotation(
  Placement(visible = true, transformation(origin = {123, -73}, extent = {{-23, -23}, {23, 23}}, rotation = 0), iconTransformation(origin = {108, -90}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealInput inputAccel annotation(
    Placement(transformation(origin = {-110, 2}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {-90, 10}, extent = {{-20, -20}, {20, 20}})));
equation
  connect(x, simple_car.x) annotation(
    Line(points = {{120, 72}, {-1, 72}, {-1, 21}, {7, 21}}, color = {0, 0, 127}));
  connect(simple_car.speed, speed) annotation(
    Line(points = {{7, 8}, {120, 8}}, color = {0, 0, 127}));
  connect(inputAccel, simple_car.accel) annotation(
    Line(points = {{-110, 2}, {-40, 2}, {-40, 8}}, color = {0, 0, 127}));
  connect(simple_car.accel_out, outputAccel) annotation(
    Line(points = {{8, 6}, {124, 6}, {124, -72}}, color = {0, 0, 127}));
  annotation(
    uses(Modelica(version = "4.0.0")),
    Diagram);
end LeadCarPrediction;