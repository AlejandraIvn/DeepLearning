
import { useState } from 'react'
import './App.css'

function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)

  const handleFile = (file) => {
  if (file && ['image/jpeg', 'image/png'].includes(file.type)) {
    setImage(file)
    setPreview(URL.createObjectURL(file))
  }
}

const handleImageChange = (event) => {
  handleFile(event.target.files[0])
}

const handleDrop = (event) => {
  event.preventDefault()
  handleFile(event.dataTransfer.files[0])
}

  return (
    <main className="app">
      <header className="header">
        <div className="brand">
          <div className="brand-icon">◉</div>
          <div>
            <h1>GlaucoAI</h1>
            <p>Inteligencia artificial para la salud visual</p>
          </div>
        </div>
      </header>

      <section className="main-content">
        <div className="welcome">
          <span className="welcome-tag">TU VISIÓN NOS IMPORTA</span>
          <h2>¡Hola! 👋</h2>
          <h3>Exploremos la salud visual.</h3>
          <p>
            Carga una imagen de fondo de ojo para analizarla
            mediante nuestro modelo de inteligencia artificial.
          </p>
        </div>

        <section className="upload-card">
          <div className="upload-heading">
            <h2>Analiza una imagen</h2>
            <p>Selecciona una imagen para comenzar.</p>
          </div>

          <label
  className="upload-area"
  onDragOver={(event) => event.preventDefault()}
  onDrop={handleDrop}
>
            {preview ? (
              <img src={preview} alt="Vista previa del fondo de ojo" />
            ) : (
              <>
                <span className="upload-icon">↑</span>
                <strong>Arrastra tu imagen aquí</strong>
                <span>o haz clic para buscarla en tu equipo</span>
                <small>Formatos admitidos: JPG y PNG</small>
              </>
            )}

            <input
              type="file"
              accept="image/png, image/jpeg"
              onChange={handleImageChange}
            />
          </label>

          {image && (
  <div className="file-info">
    <p className="file-name">
      Imagen seleccionada: {image.name}
    </p>

    <button
      className="change-image"
      onClick={() => {
        setImage(null)
        setPreview(null)
      }}
    >
      Cambiar imagen
    </button>
  </div>
)}

          <button className="analyze-button" disabled={!image}>
            Analizar imagen
          </button>

          <p className="disclaimer">
            Esta herramienta tiene fines académicos y experimentales.
            No sustituye una valoración médica profesional.
          </p>
          <div className="result-placeholder">
            <div className="result-icon">✦</div>

            <h3>Tu resultado aparecerá aquí</h3>

            <p>
              Una vez analizada la imagen, podrás consultar
              la clasificación y la probabilidad estimada
              por nuestro modelo.
            </p>
          </div>
        </section>
      </section>

      <footer className="footer">
        GlaucoAI · Proyecto de Deep Learning
      </footer>
    </main>
  )
}

export default App
