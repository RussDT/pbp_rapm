import { useState, useMemo, useEffect } from 'react'
import Papa from 'papaparse'
import './App.css'

function App() {
  const [files, setFiles] = useState([])
  const [data, setData] = useState([])
  const [headers, setHeaders] = useState([])
  const [currentFile, setCurrentFile] = useState(null)
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' })
  const [searchTerm, setSearchTerm] = useState('')

  useEffect(() => {
    fetch('/data/manifest.json')
      .then(res => res.json())
      .then(setFiles)
      .catch(console.error)
  }, [])

  const loadFile = async (fileName) => {
    const res = await fetch(`/data/${fileName}`)
    const text = await res.text()
    const { data, meta } = Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true })
    setHeaders(meta.fields || [])
    setData(data)
    setCurrentFile(fileName)
    setSortConfig({ key: null, direction: 'asc' })
    setSearchTerm('')
  }

  const handleSort = (key) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc'
    }))
  }

  const sortedData = useMemo(() => {
    if (!sortConfig.key) return data
    return [...data].sort((a, b) => {
      const aVal = a[sortConfig.key], bVal = b[sortConfig.key]
      if (aVal == null) return 1
      if (bVal == null) return -1
      const cmp = typeof aVal === 'number' ? aVal - bVal : String(aVal).localeCompare(String(bVal))
      return sortConfig.direction === 'asc' ? cmp : -cmp
    })
  }, [data, sortConfig])

  const filteredData = useMemo(() => {
    if (!searchTerm) return sortedData
    const term = searchTerm.toLowerCase()
    return sortedData.filter(row => Object.values(row).some(v => String(v).toLowerCase().includes(term)))
  }, [sortedData, searchTerm])

  return (
    <div className="app">
      <div className="sidebar">
        <h2>CSV Explorer</h2>
        <div className="folder-path">nba_pipeline/results</div>
        <div className="file-list">
          {files.map(f => (
            <div key={f} className={`file-item ${currentFile === f ? 'active' : ''}`} onClick={() => loadFile(f)}>
              {f}
            </div>
          ))}
        </div>
      </div>

      <div className="main">
        {currentFile ? (
          <>
            <div className="toolbar">
              <h3>{currentFile}</h3>
              <span className="row-count">{filteredData.length} rows</span>
              <input
                type="text"
                placeholder="Search..."
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                className="search-input"
              />
            </div>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    {headers.map(h => (
                      <th key={h} onClick={() => handleSort(h)}>
                        {h} {sortConfig.key === h ? (sortConfig.direction === 'asc' ? '↑' : '↓') : '↕'}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredData.map((row, i) => (
                    <tr key={i}>
                      {headers.map(h => (
                        <td key={h}>
                          {typeof row[h] === 'number' && !Number.isInteger(row[h]) ? row[h].toFixed(2) : row[h]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <div className="empty-state">Click a file to view</div>
        )}
      </div>
    </div>
  )
}

export default App
