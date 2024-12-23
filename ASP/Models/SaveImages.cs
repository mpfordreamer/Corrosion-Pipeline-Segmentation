using Microsoft.AspNetCore.Http;

namespace UjiCoba.Models
{
    public class SaveImages
    {
        public string PredictionType { get; set; }
        public List<string> ImageFileNames { get; set; } // Jika Anda ingin menyimpan nama file gambar

    }
    public class DownloadRequest
    {
        public string PredictionType { get; set; }
        public List<string> Files { get; set; }
    }   

}
