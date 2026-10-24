-- List user-defined tables.
SELECT name AS table_name
FROM sqlite_master
WHERE type = 'table'
  AND name NOT LIKE 'sqlite_%'
ORDER BY name;

-- Show inventory columns and their types.
PRAGMA table_info(inventory);

-- Count inventory items.
SELECT COUNT(*) AS total_items
FROM inventory;

-- Preview the first 20 items.
SELECT
    item_id,
    category,
    item_name,
    price,
    stock_quantity,
    supplier
FROM inventory
ORDER BY item_id
LIMIT 20;
