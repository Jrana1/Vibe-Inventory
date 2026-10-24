import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { AlertTriangle, ArrowLeft, ArrowRight, BarChart3, Check, ChevronDown, CircleDollarSign, CloudOff, Minus, Package, Plus, RefreshCw, Save, Search, SlidersHorizontal, Truck, X } from "lucide-react";
import "./styles.css";
import "./inventory-controls.css";
import "./inventory-actions.css";
import "./dashboard-insights.css";

type Item = { item_id: number; category: string; item_name: string; price: number | null; stock_quantity: number; supplier: string };
type Dashboard = { totals: { items: number; units: number; value: number; average_price: number; low_stock: number; out_of_stock: number; data_issues: number; attention_count: number; unpriced_items: number }; categories: { category: string; item_count: number; units: number; value: number }[]; suppliers: { supplier: string; item_count: number; units: number; value: number }[]; attention: { item_id: number; item_name: string; stock_quantity: number; price: number; supplier: string }[] };
type InventoryResponse = { items: Item[]; total: number; page: number; page_size: number; categories: string[]; suppliers: string[] };

const API = import.meta.env.VITE_API_URL || "http://localhost:8000/api";
const money = (value: number | null) => value === null ? "—" : new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value);
const compact = (value: number) => new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
const displayItemName = (name: string) => name.trim().toLowerCase() === "null" ? "Unnamed item" : name;

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, { ...options, headers: { "Content-Type": "application/json", ...options?.headers } });
  if (!response.ok) throw new Error((await response.json()).detail ?? "Request failed");
  return response.json();
}

function App() {
  const [inventory, setInventory] = useState<InventoryResponse | null>(() => JSON.parse(localStorage.getItem("vibeworx-inventory") || "null"));
  const [dashboard, setDashboard] = useState<Dashboard | null>(() => JSON.parse(localStorage.getItem("vibeworx-dashboard") || "null"));
  const [offline, setOffline] = useState(false);
  const [loading, setLoading] = useState(!inventory);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("");
  const [supplier, setSupplier] = useState("");
  const [stockStatus, setStockStatus] = useState("");
  const [sort, setSort] = useState("name");
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<number[]>([]);
  const [priceAdjustment, setPriceAdjustment] = useState("5");
  const [adjustmentType, setAdjustmentType] = useState<"percent" | "fixed">("percent");
  const [saved, setSaved] = useState("");

  useEffect(() => {
    if (!saved) return;
    const timeout = window.setTimeout(() => setSaved(""), 5000);
    return () => window.clearTimeout(timeout);
  }, [saved]);

  const refresh = async () => {
    setLoading(true); setError("");
    try {
      const params = new URLSearchParams({ search: query, category, supplier, stock_status: stockStatus, sort, page: String(page), page_size: "12" });
      const [nextInventory, nextDashboard] = await Promise.all([request<InventoryResponse>(`/inventory?${params}`), request<Dashboard>("/dashboard")]);
      setInventory(nextInventory); setDashboard(nextDashboard); setOffline(false);
      localStorage.setItem("vibeworx-inventory", JSON.stringify(nextInventory)); localStorage.setItem("vibeworx-dashboard", JSON.stringify(nextDashboard));
    } catch (err) { setOffline(true); setError(err instanceof Error ? err.message : "Unable to reach the API"); }
    finally { setLoading(false); }
  };

  useEffect(() => { void refresh(); }, [query, category, supplier, stockStatus, sort, page]);
  const allSelected = Boolean(inventory?.items.length && inventory.items.every((item) => selected.includes(item.item_id)));
  const pageCount = Math.max(1, Math.ceil((inventory?.total || 0) / (inventory?.page_size || 12)));
  const maxCategoryValue = useMemo(() => Math.max(...(dashboard?.categories.map((item) => item.value) || [1])), [dashboard]);
  const maxSupplierValue = useMemo(() => Math.max(...(dashboard?.suppliers.map((item) => item.value) || [1])), [dashboard]);

  const updateQuantity = async (item: Item, value: string) => {
    const quantity = Number(value);
    if (!Number.isInteger(quantity) || quantity < 0) return;
    try { await request(`/inventory/${item.item_id}/quantity`, { method: "PATCH", body: JSON.stringify({ stock_quantity: quantity }) }); setSaved(`Quantity saved for ${item.item_name}`); await refresh(); }
    catch (err) { setError(err instanceof Error ? err.message : "Could not save quantity"); throw err; }
  };

  const updatePrices = async () => {
    if (!selected.length) return;
    try { await request("/inventory/bulk-price", { method: "PATCH", body: JSON.stringify({ item_ids: selected, adjustment_type: adjustmentType, adjustment: Number(priceAdjustment) }) }); setSelected([]); setSaved(`${selected.length} prices updated`); await refresh(); }
    catch (err) { setError(err instanceof Error ? err.message : "Could not update prices"); }
  };

  return <div className="app-shell">
    <header className="topbar"><div className="brand"><div className="brand-mark">VW</div><div><strong>VibeWorx</strong><span>WHOLESALE OPERATIONS</span></div></div><div className="topbar-right"><div className={`connection ${offline ? "offline" : ""}`}><span className="connection-dot" />{offline ? "Offline cache" : "Live inventory"}</div><button className="icon-button" title="Refresh data" onClick={() => void refresh()}><RefreshCw size={17} className={loading ? "spin" : ""} /></button></div></header>
    <main className="content">
      <section className="page-heading"><div><p className="eyebrow">INVENTORY CONTROL / OVERVIEW</p><h1>Inventory overview</h1><p className="lede">Your warehouse position at a glance. Keep stock moving.</p></div><div className="date-chip"><span>LAST SYNC</span><strong>{offline ? "Using saved data" : new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</strong></div></section>
      {error && <div className="notice error"><AlertTriangle size={17} />{error}<button onClick={() => setError("")}><X size={15} /></button></div>}
      {offline && <div className="offline-banner"><CloudOff size={17} /><span>Showing the last successful sync. Editing is disabled until the connection returns.</span></div>}
      <section className="kpi-grid">{[
        ["Inventory value", dashboard ? money(dashboard.totals.value) : "--", "At retail cost", CircleDollarSign, "mint"],
        ["Units on hand", dashboard ? compact(dashboard.totals.units) : "--", `${dashboard?.totals.items || 0} SKUs tracked`, Package, "coral"],
        ["Low stock", dashboard ? dashboard.totals.low_stock.toString() : "--", `${dashboard?.totals.out_of_stock || 0} out of stock · ${dashboard?.totals.data_issues || 0} data issues`, AlertTriangle, "amber"],
        ["Unpriced SKUs", dashboard?.totals.unpriced_items !== undefined ? dashboard.totals.unpriced_items.toString() : "--", "Cannot be bulk repriced", CircleDollarSign, "blue"],
      ].map(([label, value, sub, Icon, color]) => <div className="kpi" key={label as string}><div className={`kpi-icon ${color}`}><Icon size={19} /></div><div><span>{label as string}</span><strong>{value as string}</strong><small>{sub as string}</small></div></div>)}</section>
      <section className="insights-grid"><div className="panel chart-panel"><div className="panel-heading"><div><p className="eyebrow">CAPITAL ALLOCATION</p><h2>Value by category</h2></div><BarChart3 size={20} className="muted-icon" /></div><div className="bars">{dashboard?.categories.map((item) => <div className="bar-row" key={item.category}><span title={item.category}>{item.category}</span><div className="bar-track"><div className="bar-fill" style={{ width: `${(item.value / maxCategoryValue) * 100}%` }} /></div><strong>{money(item.value)}</strong></div>)}</div></div><div className="panel attention-panel"><div className="panel-heading"><div><p className="eyebrow">ACTION QUEUE</p><h2>Needs attention</h2></div><span className="count-badge">{dashboard?.totals.attention_count || 0}</span></div><div className="attention-list">{dashboard?.attention.map((item) => <div className="attention-item" key={item.item_id}><div className="warning-dot"><AlertTriangle size={14} /></div><div><strong>{displayItemName(item.item_name)}</strong><span>{item.supplier}</span></div><b>{item.stock_quantity < 0 ? "Invalid stock" : `${item.stock_quantity} left`}</b></div>)}</div></div></section>
      <section className="supplier-section panel"><div className="panel-heading"><div><p className="eyebrow">SUPPLIER EXPOSURE</p><h2>Inventory value by supplier</h2></div><Truck size={20} className="muted-icon" /></div><div className="supplier-bars">{dashboard?.suppliers.map((item) => <div className="bar-row" key={item.supplier}><span title={item.supplier}>{item.supplier}</span><div className="bar-track"><div className="bar-fill supplier-fill" style={{ width: `${(item.value / maxSupplierValue) * 100}%` }} /></div><strong>{money(item.value)}</strong></div>)}</div></section>
      <section className="inventory-section"><div className="section-heading"><div><p className="eyebrow">CATALOG / {inventory?.total || 0} RESULTS</p><h2>Inventory register</h2></div><div className="filter-label"><SlidersHorizontal size={16} /> Smart filters</div></div>
        {saved && <div className="notice success action-confirmation"><span className="confirmation-icon"><Check size={16} /></span><div><strong>Update complete</strong><span>{saved}</span></div><button aria-label="Dismiss update confirmation" onClick={() => setSaved("")}><X size={15} /></button></div>}
        <div className="filters"><label className="search-field"><Search size={17} /><input placeholder="Search item, category, supplier" value={query} onChange={(e) => { setPage(1); setQuery(e.target.value); }} /></label><FilterSelect value={category} onChange={(value) => { setPage(1); setCategory(value); }} placeholder="All categories" options={inventory?.categories || []} /><FilterSelect value={supplier} onChange={(value) => { setPage(1); setSupplier(value); }} placeholder="All suppliers" options={inventory?.suppliers || []} /><FilterSelect value={stockStatus} onChange={(value) => { setPage(1); setStockStatus(value); }} placeholder="Any stock level" options={["in-stock", "low-stock", "out-of-stock"]} />{(query || category || supplier || stockStatus) && <button className="reset-button" onClick={() => { setQuery(""); setCategory(""); setSupplier(""); setStockStatus(""); setPage(1); }}>Reset filters</button>}</div>
        {selected.length === 0 && <p className="selection-help">Select items using the checkboxes to update multiple prices together.</p>}
        {selected.length > 0 && <div className="bulk-bar"><div className="bulk-heading"><strong>{selected.length} item{selected.length === 1 ? "" : "s"} selected</strong><span>Update their prices together</span></div><label className="bulk-field"><span>Change price by</span><select value={adjustmentType} onChange={(e) => setAdjustmentType(e.target.value as "percent" | "fixed")}><option value="percent">Percentage</option><option value="fixed">Fixed amount</option></select></label><label className="bulk-field amount-field"><span>Amount</span><div><input type="number" value={priceAdjustment} onChange={(e) => setPriceAdjustment(e.target.value)} aria-label="Price adjustment amount" /><b>{adjustmentType === "percent" ? "%" : "$"}</b></div></label><button className="primary-button" onClick={() => void updatePrices()} disabled={offline || priceAdjustment === "" || Number.isNaN(Number(priceAdjustment))}>Update {selected.length} price{selected.length === 1 ? "" : "s"}</button><button className="clear-button" onClick={() => setSelected([])}>Clear selection</button></div>}
        <div className="table-wrap"><table><thead><tr><th><input type="checkbox" aria-label="Select all visible items" checked={allSelected} onChange={() => setSelected(allSelected ? [] : (inventory?.items.map((item) => item.item_id) || []))} /></th><th>Item <button className="sort-button" onClick={() => { setPage(1); setSort("name"); }}><ChevronDown size={14} /></button></th><th>Category</th><th>Supplier</th><th>Unit price</th><th>Stock quantity</th><th>Status</th></tr></thead><tbody>{inventory?.items.map((item) => <tr key={item.item_id}><td><input type="checkbox" aria-label={`Select ${displayItemName(item.item_name)}`} checked={selected.includes(item.item_id)} onChange={() => setSelected((current) => current.includes(item.item_id) ? current.filter((id) => id !== item.item_id) : [...current, item.item_id])} /></td><td><strong className="item-name">{displayItemName(item.item_name)}</strong><small className="item-id">SKU-{String(item.item_id).padStart(4, "0")}</small></td><td><span className="category-tag">{item.category}</span></td><td>{item.supplier}</td><td className="price">{money(item.price)}</td><td><QuantityEditor item={item} offline={offline} onSave={(value) => updateQuantity(item, value)} /></td><td><Status quantity={item.stock_quantity} /></td></tr>)}</tbody></table>{!inventory?.items.length && <div className="empty">No inventory matches these filters.</div>}</div>
        <div className="table-footer"><span>Showing {inventory?.total ? ((page - 1) * (inventory.page_size || 12)) + 1 : 0}-{Math.min(page * (inventory?.page_size || 12), inventory?.total || 0)} of {inventory?.total || 0}</span><div className="pagination"><button className="page-button" disabled={page <= 1 || loading} onClick={() => setPage((current) => current - 1)}><ArrowLeft size={14} />Previous</button><span>Page {page} of {pageCount}</span><button className="page-button" disabled={page >= pageCount || loading} onClick={() => setPage((current) => current + 1)}>Next<ArrowRight size={14} /></button></div></div>
      </section>
    </main>
    <footer><Truck size={16} /> VibeWorx warehouse network <span>•</span> Data syncs automatically when connected</footer>
  </div>;
}

function FilterSelect({ value, onChange, placeholder, options }: Readonly<{ value: string; onChange: (value: string) => void; placeholder: string; options: string[] }>) { return <label className="select-field"><select value={value} onChange={(e) => onChange(e.target.value)}><option value="">{placeholder}</option>{options.map((option) => <option key={option} value={option}>{option}</option>)}</select><ChevronDown size={15} /></label>; }
function Status({ quantity }: Readonly<{ quantity: number }>) {
  let label = "In stock";
  let statusClass = "in";
  if (quantity < 0) { label = "Data issue"; statusClass = "out"; }
  else if (quantity === 0) { label = "Out of stock"; statusClass = "out"; }
  else if (quantity < 50) { label = "Low stock"; statusClass = "low"; }
  return <span className={`status ${statusClass}`}><span />{label}</span>;
}

function QuantityEditor({ item, offline, onSave }: Readonly<{ item: Item; offline: boolean; onSave: (value: string) => Promise<void> }>) {
  const [value, setValue] = useState(String(item.stock_quantity));
  const [saving, setSaving] = useState(false);
  const changed = value !== String(item.stock_quantity);

  useEffect(() => { setValue(String(item.stock_quantity)); }, [item.stock_quantity]);

  const save = async () => {
    if (!changed || saving) return;
    setSaving(true);
    try { await onSave(value); }
    catch { }
    finally { setSaving(false); }
  };

  const changeBy = (delta: number) => setValue(String(Math.max(0, (Number(value) || 0) + delta)));

  return <div className="quantity-editor"><input className="quantity-input" aria-label={`Quantity for ${item.item_name}`} type="number" min="0" value={value} disabled={offline || saving} onChange={(event) => setValue(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void save(); }} /><div className="quantity-stepper"><button className="quantity-decrease" type="button" aria-label={`Decrease quantity for ${item.item_name}`} title="Decrease by 1" disabled={offline || saving || Number(value) <= 0} onClick={() => changeBy(-1)}><Minus size={12} /></button><button className="quantity-increase" type="button" aria-label={`Increase quantity for ${item.item_name}`} title="Increase by 1" disabled={offline || saving} onClick={() => changeBy(1)}><Plus size={12} /></button></div><button className="quantity-save" aria-label={`Save quantity for ${item.item_name}`} title="Save quantity" disabled={offline || saving || !changed} onClick={() => void save()}><Save size={14} /></button></div>;
}

export default App;

createRoot(document.getElementById("root")!).render(<App />);
