<?php
// Veritabanına bağlanma
$servername = "localhost";
$username = "username";
$password = "password";
$dbname = "myDB";

$conn = new mysqli($servername, $username, $password, $dbname);
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Verileri seçme
$sql = "SELECT text FROM myTable";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    // Verileri işleme
    while($row = $result->fetch_assoc()) {
        $text = $row["text"];
        // Metin işleme işlemleri
    }
} else {
    echo "0 results";
}
$conn->close();
?>

<?php
require 'vendor/autoload.php';



