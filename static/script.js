window.initMap = function () {
  const nepalCenter = { lat: 28.3949, lng: 84.1240 };

  const map = new google.maps.Map(document.getElementById("map"), {
    center: nepalCenter,
    zoom: 6,
    mapId: "2458abfeb3ee3012df74a013",
    mapTypeId: 'hybrid'  // <-- This makes the map show satellite imagery
  });

  let marker = null;

  // Read values from hidden inputs
  const latInput = document.getElementById("latitude").value;
  const lngInput = document.getElementById("longitude").value;

  // Parse values safely
  const lat = parseFloat(latInput);
  const lng = parseFloat(lngInput);

  // Only if both lat and lng are valid numbers
  if (!isNaN(lat) && !isNaN(lng)) {
    const position = { lat, lng };
    map.setCenter(position);
    map.setZoom(12);

    marker = new google.maps.marker.AdvancedMarkerElement({
      map,
      position,
      title: "Selected Location",
    });
  }

  map.addListener("click", (e) => {
    const clickedLat = e.latLng.lat();
    const clickedLng = e.latLng.lng();

    // Update input fields
    document.getElementById("latitude").value = clickedLat.toFixed(6);
    document.getElementById("longitude").value = clickedLng.toFixed(6);

    const newPosition = { lat: clickedLat, lng: clickedLng };

    if (marker) {
      marker.position = newPosition;
    } else {
      marker = new google.maps.marker.AdvancedMarkerElement({
        map,
        position: newPosition,
        title: "Selected Location",
      });
    }

    map.setZoom(12);
    map.panTo(newPosition);
  });
};
