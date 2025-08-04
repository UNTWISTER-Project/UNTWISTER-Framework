model LeadAccelerationFromCurrentTimestamp
  Modelica.Blocks.Interfaces.RealOutput acceleration annotation(
    Placement(visible = true, transformation(origin = {120, 8}, extent = {{-20, -20}, {20, 20}}, rotation = 0), iconTransformation(origin = {106, -74}, extent = {{-10, -10}, {10, 10}}, rotation = 0))
  );
  Dummy dummy annotation(
    Placement(transformation(origin = {4, 8}, extent = {{-10, -10}, {10, 10}})));
  CustomSignalFromTimestamp customSignalFromTimestamp annotation(
    Placement(transformation(origin = {-76, 16}, extent = {{-10, -10}, {10, 10}})));
equation
  connect(dummy.accel_out, acceleration) annotation(
    Line(points = {{14, 8}, {120, 8}}, color = {0, 0, 127}));
  connect(customSignalFromTimestamp.acc, dummy.accel) annotation(
    Line(points = {{-66, 18}, {-4, 18}, {-4, 10}}, color = {0, 0, 127}));
  annotation(
    uses(Modelica(version = "4.0.0")),
    Diagram);
end LeadAccelerationFromCurrentTimestamp;