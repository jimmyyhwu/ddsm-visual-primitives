CREATE TABLE IF NOT EXISTS doctor (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS classification (
  id INTEGER PRIMARY KEY,
  description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS net (
  id TEXT PRIMARY KEY,
  net TEXT NOT NULL,
  filename TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS unit_annotation (
  unit_id INTEGER NOT NULL,
  net_id TEXT NOT NULL,
  doctor_id INTEGER NOT NULL,
  threshold REAL,
  descriptions TEXT,
  filter INTEGER,
  FOREIGN KEY(net_id) REFERENCES net(id),
  FOREIGN KEY(doctor_id) REFERENCES doctor(id),
  PRIMARY KEY(unit_id, net_id, doctor_id)
);

CREATE TABLE IF NOT EXISTS image (
  id INTEGER PRIMARY KEY,
  image_path TEXT NOT NULL,
  ground_truth INTEGER NOT NULL,
  FOREIGN KEY(ground_truth) REFERENCES classification(id)
);

CREATE TABLE IF NOT EXISTS patch (
  id INTEGER PRIMARY KEY,
  x INTEGER NOT NULL,
  y INTEGER NOT NULL,
  image_path TEXT NOT NULL,
  image_id INTEGER NOT NULL,
  ground_truth INTEGER NOT NULL,
  FOREIGN KEY(image_id) REFERENCES  image(id),
  FOREIGN KEY(ground_truth) REFERENCES classification(id)
);

CREATE TABLE IF NOT EXISTS image_unit_activation (
  net_id TEXT NOT NULL,
  patch_id INTEGER NOT NULL,
  unit_id INTEGER NOT NULL,
  activation REAL NOT NULL,
  rank INTEGER NOT NULL,
  FOREIGN KEY(net_id) REFERENCES net(id),
  FOREIGN KEY(patch_id) REFERENCES patch(id),
  PRIMARY KEY(net_id, patch_id, unit_id)
);

CREATE TABLE IF NOT EXISTS patch_unit_activation (
  net_id TEXT NOT NULL,
  patch_id INTEGER NOT NULL,
  unit_id INTEGER NOT NULL,
  activation REAL NOT NULL,
  FOREIGN KEY(net_id) REFERENCES net(id),
  FOREIGN KEY(patch_id) REFERENCES patch(id),
  PRIMARY KEY(net_id, patch_id, unit_id)
);

CREATE TABLE IF NOT EXISTS result (
  net_id TEXT NOT NULL,
  patch_id INTEGER NOT NULL,
  class INTEGER NOT NULL,
  FOREIGN KEY(net_id) REFERENCES net(id),
  FOREIGN KEY(patch_id) REFERENCES patch(id),
  FOREIGN KEY(class) REFERENCES classification(id),
  PRIMARY KEY(net_id, patch_id)
);

INSERT INTO classification (
  id,
  description
) VALUES (
  0, "no finding"
), (
  1, "benign"
), (
  2, "malignant"
);
