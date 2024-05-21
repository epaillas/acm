
 
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Periodic_3_Delaunay_triangulation_traits_3.h>
#include <CGAL/Periodic_3_Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/natural_neighbor_coordinates_3.h>

#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Sphere_3.h>
#include <CGAL/Tetrahedron_3.h>
#include <CGAL/Triangle_3.h>


#include <vector>
#include <set>
#include <cassert>
#include <algorithm>
#include <limits>
#include "omp.h"
#include <math.h>
#include <iterator>

#define DOUBLE_MAX std::numeric_limits<double>::max();
#define DOUBLE_MIN std::numeric_limits<double>::min();

// Traits and triangulation data structures
typedef CGAL::Exact_predicates_exact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_3<size_t,K> Vertex_base_info;
typedef CGAL::Triangulation_data_structure_3<Vertex_base_info, CGAL::Delaunay_triangulation_cell_base_3<K>>  TriangulationDS;
typedef CGAL::Periodic_3_Delaunay_triangulation_traits_3<K> P3Traits;
typedef CGAL::Periodic_3_Delaunay_triangulation_3<P3Traits> PDelaunay;
typedef CGAL::Delaunay_triangulation_3<K> Delaunay;
typedef CGAL::Delaunay_triangulation_3<K,TriangulationDS> DelaunayInfo;

// Iterators
typedef PDelaunay::Periodic_tetrahedron_iterator periodic_tetrahedra;
typedef Delaunay::Finite_cells_iterator finite_cells;
typedef Delaunay::Finite_vertices_iterator delaunay_vertices;
typedef Delaunay::Vertex_handle vertex_handle;
typedef Delaunay::Cell_handle cell_handle;
typedef DelaunayInfo::Finite_cells_iterator finite_cells_info;

// Geometric object types
typedef PDelaunay::Iso_cuboid Iso_cuboid;
typedef CGAL::Point_3<K> Point_3;
typedef CGAL::Sphere_3<K> Sphere_3;
typedef CGAL::Tetrahedron_3<K> Tetrahedron_3;
typedef CGAL::Triangle_3<K> Triangle_3;
typedef Delaunay::Point Point;
typedef PDelaunay::Point PPoint;
typedef K::FT FT;


struct VertexList{
    std::vector<size_t> v;
};

class DelaunayOutput{
    public:
        std::vector<double> x;
        std::vector<double> y;
        std::vector<double> z;
        std::vector<double> r;
        std::vector<double> volume;
        std::vector<double> area;
        std::vector<double> dtfe;
        std::vector<double> weights;
        std::vector<double> selection;
        std::vector<size_t> vertices[4];
        size_t n_simplices;
        

};


double tetrahedron_area(Tetrahedron_3 tetrahedron){

    double tot_area = 0;
    int i,j,k;
    for (i = 0; i < 4; i++){
        for (j = i+1; j < 4; j++){
            for (k = j+1; k < 4; k++){
                Triangle_3 facet = Triangle_3(tetrahedron[i], tetrahedron[j], tetrahedron[k]);
                tot_area += CGAL::sqrt(CGAL::to_double(facet.squared_area()));
            }
        }
    }
    

    return tot_area;

}

DelaunayOutput cdelaunay(std::vector<double> X, std::vector<double> Y, std::vector<double> Z){

    DelaunayOutput output;
    std::vector<Point_3>points;
    
    
    

    std::size_t i, n_points;
    n_points = X.size();

    for (i=0; i<n_points; i++){
        points.push_back(Point_3(X[i], Y[i], Z[i]));
    }
    
    std::cout<<"==> Number of points: "<<points.size()<<std::endl;
    std::cout<<"==> Building Delaunay Triangulation."<<std::endl;
    
    Delaunay tess(points.begin(), points.end());
    assert(tess.is_valid());
    points.clear();
    
    Point_3 simplex_vertices[4];
    Sphere_3 buffer_sphere;
    Point_3 buffer_point;

    std::cout<<"==> Number of vertices: "<<tess.number_of_vertices()<<std::endl;
    std::cout<<"==> Number of all cells: "<<tess.number_of_cells()<<std::endl;
    std::cout<<"==> Number of finite cells: "<<tess.number_of_finite_cells()<<std::endl;
    output.n_simplices = tess.number_of_finite_cells();
    
    output.x.reserve(tess.number_of_finite_cells());
    output.y.reserve(tess.number_of_finite_cells());
    output.z.reserve(tess.number_of_finite_cells());
    output.r.reserve(tess.number_of_finite_cells());
  
    std::size_t k = 0;
    
    
    
    
    for(finite_cells cell=tess.finite_cells_begin();cell!=tess.finite_cells_end();cell++) {
        for(i=0;i<4;i++){
            simplex_vertices[i] = cell->vertex(i)->point();
        }
        
        buffer_sphere = Sphere_3(simplex_vertices[0],simplex_vertices[1],simplex_vertices[2],simplex_vertices[3]);
        //buffer_point = buffer_sphere.center();
        buffer_point = cell->circumcenter();
        output.x[k] = CGAL::to_double(buffer_point.x());
        output.y[k] = CGAL::to_double(buffer_point.y());
        output.z[k] = CGAL::to_double(buffer_point.z());
        output.r[k] = CGAL::sqrt(CGAL::to_double(CGAL::squared_distance(buffer_point, cell->vertex(0)->point())));
        k++;
    }
    return output;

}


DelaunayOutput cdelaunay_periodic_extend(std::vector<double> X, std::vector<double> Y, std::vector<double> Z, std::vector<double> box_min, std::vector<double> box_max, double cpy_range){

    DelaunayOutput output;
    std::vector<Point_3>points;
    
    
    

    std::size_t i, n_points, j;
    n_points = X.size();

    double box_size[3];
    for(i=0;i<3;i++){
        box_size[i] = box_max[i] - box_min[i];
    }
    for (i=0; i<n_points; i++){
        points.push_back(Point_3(X[i], Y[i], Z[i]));
    }
    
    
    
    double point_density = n_points / (box_size[0] * box_size[1] * box_size[2]);
    double mean_free_path = pow(point_density, -1./3);
    cpy_range = cpy_range == 0 ? 8 * mean_free_path : cpy_range;

    std::cout << "==> Point density: " << point_density << " (h/Mpc)^3" << std::endl;
    std::cout << "==> Domain volume: " << (box_size[0] * box_size[1] * box_size[2]) << " (Mpc/h)^3" << std::endl;
    std::cout << "==> Mean free path: " << mean_free_path << " Mpc/h" << std::endl;
    std::cout << "==> Copy range: " << cpy_range << " Mpc/h" << std::endl;

    size_t size = points.size();
    std::cout<<"==> Number of points: "<<points.size()<<std::endl<<std::endl;
    size_t point_count = n_points;
    std::cout<<"Duplicating boundaries for periodic condition"<<std::endl;
    for(i=0;i<size;i++)
        if(points[i].x()<box_min[0]+cpy_range){
            points.push_back(Point_3(points[i].x()+box_size[0],points[i].y(),points[i].z()));
        }
    size=points.size();
    for(i=0;i<size;i++) 
        if(points[i].x()>=box_max[0]-cpy_range && points[i].x()<box_max[0]){
            points.push_back(Point_3(points[i].x()-box_size[0],points[i].y(),points[i].z()));
        }
    size=points.size();
    for(i=0;i<size;i++) 
        if(points[i].y()<box_min[1]+cpy_range){
            points.push_back(Point_3(points[i].x(),points[i].y()+box_size[1],points[i].z()));
        }
    size=points.size();
    for(i=0;i<size;i++) 
        if(points[i].y()>=box_max[1]-cpy_range && points[i].y()<box_max[1]){
            points.push_back(Point_3(points[i].x(),points[i].y()-box_size[1],points[i].z()));
        }
    size=points.size();
    for(i=0;i<size;i++) 
        if(points[i].z()<box_min[2]+cpy_range){
            points.push_back(Point_3(points[i].x(),points[i].y(),points[i].z()+box_size[2]));
        }
    size=points.size();
    for(i=0;i<size;i++) 
        if(points[i].z()>=box_max[2]-cpy_range && points[i].z()<box_max[2]){
            points.push_back(Point_3(points[i].x(),points[i].y(),points[i].z()-box_size[2]));
        }
    size=points.size();

    
    std::cout<<"==> Number of points: "<<points.size()<<std::endl;
    std::cout<<"==> Building Delaunay Triangulation."<<std::endl;
    
    Delaunay tess(points.begin(), points.end());
    assert(tess.is_valid());
    points.clear();
    
    Point_3 simplex_vertices[4];
    Sphere_3 buffer_sphere;
    Point_3 buffer_point;

    std::cout<<"==> Number of vertices: "<<tess.number_of_vertices()<<std::endl;
    std::cout<<"==> Number of all cells: "<<tess.number_of_cells()<<std::endl;
    std::cout<<"==> Number of finite cells: "<<tess.number_of_finite_cells()<<std::endl;
    output.n_simplices = tess.number_of_finite_cells();
    
    output.x.reserve(tess.number_of_finite_cells());
    output.y.reserve(tess.number_of_finite_cells());
    output.z.reserve(tess.number_of_finite_cells());
    output.r.reserve(tess.number_of_finite_cells());
  
    std::size_t k = 0;
    
    
    
    
    for(finite_cells cell=tess.finite_cells_begin();cell!=tess.finite_cells_end();cell++) {
        for(i=0;i<4;i++){
            simplex_vertices[i] = cell->vertex(i)->point();
        }
        
        buffer_sphere = Sphere_3(simplex_vertices[0],simplex_vertices[1],simplex_vertices[2],simplex_vertices[3]);
        //buffer_point = buffer_sphere.center();
        buffer_point = cell->circumcenter();
        output.x[k] = CGAL::to_double(buffer_point.x());
        output.y[k] = CGAL::to_double(buffer_point.y());
        output.z[k] = CGAL::to_double(buffer_point.z());
        output.r[k] = CGAL::sqrt(CGAL::to_double(CGAL::squared_distance(buffer_point, cell->vertex(0)->point())));
        
        

        k++;

        
    }
    return output;

}



DelaunayOutput cdelaunay_periodic(std::vector<double> X, std::vector<double> Y, std::vector<double> Z, std::vector<double> box_min, std::vector<double> box_max){

    DelaunayOutput output;

    std::vector<Point_3>points;
    
    
    

    std::size_t i, n_points, j;
    n_points = X.size();
    double box_size[3];
    for(i=0;i<3;i++){
        box_size[i] = box_max[i] - box_min[i];
    }
    for (i=0; i<n_points; i++){
        points.push_back(Point_3(X[i], Y[i], Z[i]));
        
    }

    Iso_cuboid domain(box_min[0], box_min[1], box_min[2], box_max[0], box_max[1], box_max[2]);
    for (i = 0; i < 3; i++){
        box_size[i] = box_max[i] - box_min[i];
    }
    
    std::cout<<"==> Number of points: "<<points.size()<<std::endl;
    std::cout<<"==> Building Delaunay Triangulation."<<std::endl;
    
    PDelaunay tess(points.begin(), points.end(), domain);
    assert(tess.is_valid());
    points.clear();
    
    Point_3 simplex_vertices[4];
    Sphere_3 buffer_sphere;
    Point_3 buffer_point;
    Tetrahedron_3 buffer_tetrahedron;

    std::cout<<"==> Number of vertices: "<<tess.number_of_vertices()<<std::endl;
    std::cout<<"==> Number of all cells: "<<tess.number_of_cells()<<std::endl;
    std::cout<<"==> Number of finite cells: "<<tess.number_of_finite_cells()<<std::endl;
    output.n_simplices = tess.number_of_finite_cells();
    
    output.x.reserve(tess.number_of_finite_cells());
    output.y.reserve(tess.number_of_finite_cells());
    output.z.reserve(tess.number_of_finite_cells());
    output.r.reserve(tess.number_of_finite_cells());
  
    std::size_t k = 0;
    
    
    
    
    for(periodic_tetrahedra cell=tess.periodic_tetrahedra_begin();cell!=tess.periodic_tetrahedra_end();cell++) {
        
        buffer_tetrahedron = tess.construct_tetrahedron(*cell);
        buffer_point = CGAL::circumcenter(buffer_tetrahedron);
        output.x[k] = CGAL::to_double(buffer_point.x());
        output.y[k] = CGAL::to_double(buffer_point.y());
        output.z[k] = CGAL::to_double(buffer_point.z());
        output.r[k] = CGAL::sqrt(CGAL::to_double(CGAL::squared_distance(buffer_point, buffer_tetrahedron[0])));
        //if (output.x[k] < box_min[0]) output.x[k] += box_size[0];
        //else if (output.x[k] >= box_max[0]) output.x[k] -= box_size[0];

        //if (output.y[k] < box_min[1]) output.y[k] += box_size[1];
        //else if (output.y[k] >= box_max[1]) output.y[k] -= box_size[1];

        //if (output.z[k] < box_min[2]) output.z[k] += box_size[2];
        //else if (output.z[k] >= box_max[2]) output.z[k] -= box_size[2];
        
        k++;

        
    }
    return output;

}

DelaunayOutput cdelaunay_full(std::vector<double> X, std::vector<double> Y, std::vector<double> Z){

    DelaunayOutput output;
    std::vector< std::pair<Point,size_t> >points;
    
    
    

    std::size_t i, n_points;
    n_points = X.size();

    for (i=0; i<n_points; i++){
        points.push_back(std::make_pair(Point_3(X[i], Y[i], Z[i]),i));
    }
    
    std::cout<<"==> Number of points: "<<points.size()<<std::endl;
    std::cout<<"==> Building Delaunay Triangulation."<<std::endl;
    
    DelaunayInfo tess(points.begin(), points.end());
    assert(tess.is_valid());
    points.clear();
    
    Point_3 simplex_vertices[4];
    Sphere_3 buffer_sphere;
    Point_3 buffer_point;
    Tetrahedron_3 buffer_tetrahedron;

    std::cout<<"==> Number of vertices: "<<tess.number_of_vertices()<<std::endl;
    std::cout<<"==> Number of all cells: "<<tess.number_of_cells()<<std::endl;
    std::cout<<"==> Number of finite cells: "<<tess.number_of_finite_cells()<<std::endl;
    output.n_simplices = tess.number_of_finite_cells();
    
    output.x.reserve(tess.number_of_finite_cells());
    output.y.reserve(tess.number_of_finite_cells());
    output.z.reserve(tess.number_of_finite_cells());
    output.r.reserve(tess.number_of_finite_cells());
    output.volume.reserve(tess.number_of_finite_cells());
    output.area.reserve(tess.number_of_finite_cells());
    output.dtfe.reserve(tess.number_of_vertices());
  
    std::size_t k = 0;
    for(i=0; i<tess.number_of_vertices();i++){
        output.dtfe[i] = 0;
    }
    
    
    for(finite_cells_info cell=tess.finite_cells_begin();cell!=tess.finite_cells_end();cell++) {
        
        
        buffer_tetrahedron = Tetrahedron_3(cell->vertex(0)->point(),
                                            cell->vertex(1)->point(),
                                            cell->vertex(2)->point(),
                                            cell->vertex(3)->point());
        output.volume[k] = CGAL::to_double(buffer_tetrahedron.volume());
        buffer_point = CGAL::circumcenter(buffer_tetrahedron);
        output.x[k] = CGAL::to_double(buffer_point.x());
        output.y[k] = CGAL::to_double(buffer_point.y());
        output.z[k] = CGAL::to_double(buffer_point.z());
        output.r[k] = CGAL::sqrt(CGAL::to_double(CGAL::squared_distance(buffer_point, cell->vertex(0)->point())));
        for(i=0;i<4;i++){
            output.dtfe[cell->vertex(i)->info()] += output.volume[k];
            output.vertices[i].push_back(cell->vertex(i)->info());
        }
        output.area[k] = tetrahedron_area(buffer_tetrahedron);
        
        k++;
    }
    


    return output;

}


DelaunayOutput cdelaunay_periodic_full(std::vector<double> X, std::vector<double> Y, std::vector<double> Z, std::vector<double> box_min, std::vector<double> box_max, double cpy_range){

    DelaunayOutput output;
    std::vector< std::pair<Point,size_t> >points;
    
    
    
    double box_size[3];
    std::size_t i, n_points, j;
    n_points = X.size();

    
    for(i=0;i<3;i++){
        box_size[i] = box_max[i] - box_min[i];
    }
    for (i=0; i<n_points; i++){
        points.push_back(std::make_pair(Point_3(X[i], Y[i], Z[i]),i));
    }
    for (i = 0; i < 3; i++){
        box_size[i] = box_max[i] - box_min[i];
    }
    
    double point_density = n_points / (box_size[0] * box_size[1] * box_size[2]);
    double mean_free_path = pow(point_density, -1./3);
    cpy_range == 0. ? 8 * mean_free_path : cpy_range;

    std::cout << "==> Point density: " << point_density << " (h/Mpc)^3" << std::endl;
    std::cout << "==> Domain volume: " << (box_size[0] * box_size[1] * box_size[2]) << " (Mpc/h)^3" << std::endl;
    std::cout << "==> Mean free path: " << mean_free_path << " Mpc/h" << std::endl;
    std::cout << "==> Copy range: " << cpy_range << " Mpc/h" << std::endl;

    size_t size = points.size();
    std::cout<<"==> Number of points: "<<points.size()<<std::endl<<std::endl;
    size_t point_count = n_points;
    std::cout<<"Duplicating boundaries for periodic condition"<<std::endl;
    for(i=0;i<size;i++)
        if(points[i].first.x()<box_min[0]+cpy_range){
            points.push_back(std::make_pair(Point_3(points[i].first.x()+box_size[0],points[i].first.y(),points[i].first.z()), point_count++));
        }
    size=points.size();
    for(i=0;i<size;i++) 
        if(points[i].first.x()>=box_max[0]-cpy_range && points[i].first.x()<box_max[0]){
            points.push_back(std::make_pair(Point_3(points[i].first.x()-box_size[0],points[i].first.y(),points[i].first.z()), point_count++));
        }
    size=points.size();
    for(i=0;i<size;i++) 
        if(points[i].first.y()<box_min[1]+cpy_range){
            points.push_back(std::make_pair(Point_3(points[i].first.x(),points[i].first.y()+box_size[1],points[i].first.z()), point_count++));
        }
    size=points.size();
    for(i=0;i<size;i++) 
        if(points[i].first.y()>=box_max[1]-cpy_range && points[i].first.y()<box_max[1]){
            points.push_back(std::make_pair(Point_3(points[i].first.x(),points[i].first.y()-box_size[1],points[i].first.z()), point_count++));
        }
    size=points.size();
    for(i=0;i<size;i++) 
        if(points[i].first.z()<box_min[2]+cpy_range){
            points.push_back(std::make_pair(Point_3(points[i].first.x(),points[i].first.y(),points[i].first.z()+box_size[2]), point_count++));
        }
    size=points.size();
    for(i=0;i<size;i++) 
        if(points[i].first.z()>=box_max[2]-cpy_range && points[i].first.z()<box_max[2]){
            points.push_back(std::make_pair(Point_3(points[i].first.x(),points[i].first.y(),points[i].first.z()-box_size[2]), point_count++));
        }
    size=points.size();

    
    std::cout<<"==> Number of points: "<<points.size()<<std::endl;
    std::cout<<"==> Building Delaunay Triangulation."<<std::endl;
    
    DelaunayInfo tess(points.begin(), points.end());
    assert(tess.is_valid());
    points.clear();
    
    Point_3 simplex_vertices[4];
    Sphere_3 buffer_sphere;
    Point_3 buffer_point;
    Tetrahedron_3 buffer_tetrahedron;

    std::cout<<"==> Number of vertices: "<<tess.number_of_vertices()<<std::endl;
    std::cout<<"==> Number of all cells: "<<tess.number_of_cells()<<std::endl;
    std::cout<<"==> Number of finite cells: "<<tess.number_of_finite_cells()<<std::endl;
    output.n_simplices = tess.number_of_finite_cells();
    
    output.x.reserve(tess.number_of_finite_cells());
    output.y.reserve(tess.number_of_finite_cells());
    output.z.reserve(tess.number_of_finite_cells());
    output.r.reserve(tess.number_of_finite_cells());
    output.volume.reserve(tess.number_of_finite_cells());
    output.area.reserve(tess.number_of_finite_cells());
    output.dtfe.reserve(tess.number_of_vertices());
    for(i=0;i<4;i++){
            output.vertices[i].reserve(tess.number_of_finite_cells());
        }
    
  
    std::size_t k = 0;
    for(i=0; i<tess.number_of_vertices();i++){
        output.dtfe[i] = 0;
    }
    
    
    
    for(finite_cells_info cell=tess.finite_cells_begin();cell!=tess.finite_cells_end();cell++) {
        
        //std::cout<<" " << k << " " ;
        buffer_tetrahedron = Tetrahedron_3(cell->vertex(0)->point(),
                                            cell->vertex(1)->point(),
                                            cell->vertex(2)->point(),
                                            cell->vertex(3)->point());
        //std::cout<<" A " ;                                            
        output.volume[k] = CGAL::to_double(buffer_tetrahedron.volume());
        //std::cout<<" B " ;
        //buffer_point = CGAL::circumcenter(buffer_tetrahedron);
        buffer_point = cell->circumcenter();
        //std::cout<<" C " ;
        output.x[k] = CGAL::to_double(buffer_point.x());
        //std::cout<<" D " ;
        output.y[k] = CGAL::to_double(buffer_point.y());
        //std::cout<<" E " ;
        output.z[k] = CGAL::to_double(buffer_point.z());
        //std::cout<<" F " ;
        output.r[k] = CGAL::sqrt(CGAL::to_double(CGAL::squared_distance(buffer_point, cell->vertex(0)->point())));
        //std::cout<<" G " ;
        for(i=0;i<4;i++){
            output.dtfe[cell->vertex(i)->info()] += output.volume[k];
            //std::cout<< i<< "G1 " ;
            //std::cout<< " " << cell->vertex(i)->info() << " ";
            output.vertices[i][k] = (cell->vertex(i)->info());
            //std::cout<< i<< "G2 " ;
        }
        //std::cout<<" H " ;
        output.area[k] = tetrahedron_area(buffer_tetrahedron);
        //std::cout<<" I " ;
        
        k++;
    }
    


    return output;

}



int main(){
    return 0;
}

