# Decisions

## Design decisions

- **FastAPI + SQLAlchemy + SQLite:** FastAPI provides a small typed API, SQLAlchemy provides the ORM/session boundary, and the supplied SQLite database remains the source of truth. The existing schema is mapped without replacing or reseeding the provided data.
- **Backend structure:** HTTP routers, Pydantic request/response schemas, SQLAlchemy models, and service-layer query/business logic are separated so each concern is easy to test and maintain.
- **Inventory filters:** The table supports free-text search across item, category, and supplier, exact category and supplier filters, and stock states: in stock (50+), low stock (1-49), and out of stock (0). Pagination shows 12 items per page so the table remains easy to scan.
- **Quantity updates:** Quantity editing uses direct numeric entry plus adjacent `-` and `+` controls. Changes require an explicit Save action, preventing accidental updates from blur events. Negative quantities are rejected by the update API.
- **Bulk pricing:** Selected rows can receive either a percentage adjustment or a fixed-dollar adjustment. The action bar states the number of selected items and displays the correct `%` or `$` unit. The API rejects negative resulting prices and refuses to reprice records without a price.
- **Dashboard:** The dashboard focuses on metrics available in the source data: inventory value, units on hand, SKU count, low-stock count, out-of-stock count, data-quality issues, unpriced SKUs, category value, supplier value exposure, and an attention queue. Sales, turnover, and historical trend charts were intentionally not invented because the database has no sales, cost, or timestamp fields.
- **Data quality:** Literal `Null` item names are displayed as `Unnamed item`; negative quantities are displayed as `Invalid stock` and `Data issue`. The source database is not silently modified.
- **Connectivity:** The browser stores the latest successful inventory and dashboard responses in `localStorage`. If the API is unavailable, the last view remains readable and edits are disabled until the connection returns.
- **Configuration:** The frontend defaults to `http://localhost:8000/api` and supports `VITE_API_URL` through `frontend/.env.local` for another backend host or port.
- **Visual design:** The interface is intentionally dense and operational: clear status colors, explicit action labels, responsive controls, pagination, and a scan-friendly inventory table.

## Limitations and shortcuts

- There is no authentication, role management, audit log, or multi-warehouse model because these were not present in the supplied schema or challenge scope.
- Cached data is local to one browser and is not an offline write queue. Quantity and price edits require a live API connection.
- The database is updated directly through SQLite transactions. A production multi-process deployment would need a database strategy designed for concurrent writes and migrations.
- The dashboard is a current-state report only. Historical trends would require inventory snapshots or timestamped events that are not present in the source database.
- The supplied database contains a small number of invalid or incomplete records, including negative stock and missing prices. The UI labels these clearly, but does not alter the source data.
- Automated API and browser test coverage is limited to smoke checks and build validation due to the interview time constraint.

## With more time

- Add authentication, permissions, and a server-side audit trail for every inventory change.
- Add a complete automated test suite for filters, pagination, quantity updates, bulk pricing, validation failures, cached data, and responsive interactions.
- Add an offline mutation queue with conflict handling and visible sync history.
- Add inventory snapshots and event history to support stock, valuation, and supplier-exposure trends over time.
- Add configurable stock thresholds instead of hard-coding the 50-unit low-stock boundary.
- Add deployment configuration, containerized startup, database migrations, and environment-specific CORS/API settings.
