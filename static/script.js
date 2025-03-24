document.addEventListener("DOMContentLoaded", function () {
    let contactBtn = document.getElementById("contact-btn");
    if (contactBtn) {
      contactBtn.addEventListener("click", function () {
        if (navigator.geolocation) {
          navigator.geolocation.getCurrentPosition(fetchHospitals, errorHandler);
        } else {
          alert("Geolocation is not supported by your browser.");
        }
      });
    }
  });
  
  function fetchHospitals(position) {
    let lat = position.coords.latitude;
    let lon = position.coords.longitude;
    let radius = 5000;  // 5km radius
  
    let overpassURL = `https://overpass-api.de/api/interpreter?data=[out:json];node[amenity=hospital](around:${radius},${lat},${lon});out;`;
  
    fetch(overpassURL)
      .then(response => response.json())
      .then(data => {
        let hospitalList = document.getElementById("hospital-list");
        hospitalList.innerHTML = "";
  
        if (data.elements.length === 0) {
          hospitalList.innerHTML = "<li class='list-group-item'>No hospitals found nearby.</li>";
        } else {
          data.elements.forEach(hospital => {
            let hospitalName = hospital.tags.name || "Unnamed Hospital";
            let hospitalLat = hospital.lat;
            let hospitalLon = hospital.lon;
            let phone = hospital.tags.phone || "";
            let website = hospital.tags.website || hospital.tags["contact:website"] || "";
            let email = hospital.tags.email || hospital.tags["contact:email"] || "";
  
            // Navigation button
            let navLink = `<a href="https://www.google.com/maps/dir/?api=1&destination=${hospitalLat},${hospitalLon}" target="_blank" class="btn btn-primary btn-sm">Navigate</a>`;
  
            // Call button (if phone available)
            let callLink = phone ? `<a href="tel:${phone}" class="btn btn-success btn-sm">Call</a>` : "";
  
            // Website button (if available)
            let websiteLink = website ? `<a href="${website}" target="_blank" class="btn btn-info btn-sm">Website</a>` : "";
  
            // Email button (if available)
            let emailLink = email ? `<a href="mailto:${email}" class="btn btn-warning btn-sm">Email</a>` : "";
  
            // Google Search button (always present)
            let googleSearch = `<a href="https://www.google.com/search?q=${encodeURIComponent(hospitalName + ' hospital')}" target="_blank" class="btn btn-secondary btn-sm">Search on Google</a>`;
  
            let listItem = document.createElement("li");
            listItem.className = "list-group-item d-flex justify-content-between align-items-center";
            listItem.innerHTML = `
              <div><strong>${hospitalName}</strong></div>
              <div class="d-flex gap-2 justify-content-end">
                ${callLink} ${websiteLink} ${emailLink} ${googleSearch} ${navLink}
              </div>
            `;
  
            hospitalList.appendChild(listItem);
          });
        }
  
        document.getElementById("hospital-results").classList.remove("d-none");
      })
      .catch(error => {
        console.error("Error fetching hospital data:", error);
        alert("Failed to load hospital data.");
      });
  }
  
  function errorHandler(error) {
    alert("Error getting location. Please allow location access.");
  }
  