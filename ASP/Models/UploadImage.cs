using Microsoft.AspNetCore.Http;

namespace UjiCoba.Models
{
    public class UploadImage
    {
        public List<IFormFile> ImageFiles { get; set; } = new List<IFormFile>();
        public string SegmentationType { get; set; } // Add this property
        // public int TotalPixel { get; set; }
        // public int AssetPixel { get; set; }
        // public int CorrosionPixels { get; set; }
        // public string AssetOfImage { get; set; }
        // public string CorrosionOfAsset { get; set; }
    }

    public class SegmentData
    {
        public string SegmentType { get; set; }
        public List<string> ImageUrls { get; set; }

    }
}