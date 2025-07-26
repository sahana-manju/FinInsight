function addStockRow() {
    const container = document.getElementById('stocksContainer');

    const row = document.createElement('div');
    row.className = 'row mb-2 stock-row';

    const selectHTML = STOCK_OPTIONS.map(option => `<option value="${option}">${option}</option>`).join('');

    row.innerHTML = `
        <div class="col-md-6">
            <select class="form-select" name="stock_name" required>
                ${selectHTML}
            </select>
        </div>
        <div class="col-md-4">
            <input type="number" class="form-control" name="stock_percent" required min="0" max="100" step="0.01" placeholder="Allocation %">
        </div>
        <div class="col-md-2">
            <button type="button" class="btn btn-danger" onclick="removeStockRow(this)">Delete</button>
        </div>
    `;
    container.appendChild(row);
}

function removeStockRow(button) {
    const row = button.closest('.stock-row');
    if (row) row.remove();
}


