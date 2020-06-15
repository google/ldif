// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Source file for the pdb viewer program

////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

namespace gaps {}
using namespace gaps;  // NOLINT(build/namespaces)
#include "R3Graphics/R3Graphics.h"
#include "R3Quadric.h"
#include "R3QuadricSet.h"
#include "fglut/fglut.h"

////////////////////////////////////////////////////////////////////////
// Global variables
////////////////////////////////////////////////////////////////////////

// Program variables
static const char *input_quadrics_filename = NULL;
static const char *input_mesh_filename = NULL;
static const char *output_isosurface_filename = NULL;
static const char *output_image_filename = NULL;
static RNScalar background_color[3] = {1, 1, 1};
static int evaluation_method = R3_QUADRIC_SET_WEIGHTED_VALUE;
static int resolution = 256;
static int print_verbose = 0;

// GLUT variables
static int GLUTwindow = 0;
static int GLUTwindow_height = 480;
static int GLUTwindow_width = 640;
static int GLUTmouse[2] = {0, 0};
static int GLUTbutton[3] = {0, 0, 0};
static int GLUTmouse_drag = 0;
static int GLUTmodifiers = 0;

// Application variables
static R3QuadricSet *quadrics = NULL;
static R3Mesh *mesh = NULL;
static R3Viewer *viewer = NULL;
static R3Point initial_camera_origin = R3Point(0.0, 0.0, 0.0);
static R3Vector initial_camera_towards = R3Vector(0.0, 0.0, -1.0);
static R3Vector initial_camera_up = R3Vector(0.0, 1.0, 0.0);
static RNBoolean initial_camera = FALSE;
static R3Point world_origin(0, 0, 0);
static R3Point query_position(0, 0, 0);
static RNBoolean color_by_query_support = 0;
static RNScalar isolevel = -0.07;

// Display variables
static int show_isosurface_faces = 0;
static int show_isosurface_edges = 0;
static int show_quadric_ellipsoids = 1;
static int show_quadric_text = 0;
static int show_quadric_transparency = 0;
static int show_quadric_samples = 0;
static int show_mesh_faces = 1;
static int show_mesh_edges = 0;
static int show_query_position = 0;
static int show_message_text = 0;
static int show_bbox = 0;
static int show_axes = 0;

// Selection variables
static int selected_quadric_index = -1;
static int moving_query_position = 0;
static double quadric_transparency_clamp = -3.0;

////////////////////////////////////////////////////////////////////////
// I/O functions
////////////////////////////////////////////////////////////////////////

static R3QuadricSet *ReadQuadrics(const char *filename) {
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Hardwire bbox for now
  R3Box bbox(-0.6, -0.6, -0.6, 0.6, 0.6, 0.6);

  // Allocate quadrics
  R3QuadricSet *quadrics = new R3QuadricSet(bbox);
  if (!quadrics) {
    RNFail("Unable to allocate quadrics");
    return NULL;
  }

  // Read quadrics
  if (!quadrics->ReadFile(filename)) {
    RNFail("Unable to read quadrics from file %s", filename);
    return NULL;
  }

  // Set evaluation method
  quadrics->SetEvaluationMethod(evaluation_method);

  // Set resolution
  quadrics->SetResolution(resolution);

  // Print statistics
  if (print_verbose) {
    printf("Read quadrics from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Quadrics = %d\n", quadrics->NQuadrics());
    fflush(stdout);
  }

  // Return success
  return quadrics;
}

static R3Mesh *ReadMesh(const char *filename) {
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate mesh
  R3Mesh *mesh = new R3Mesh();
  if (!mesh) {
    RNFail("Unable to allocate mesh for %s\n", filename);
    return NULL;
  }

  // Read mesh from file
  if (!mesh->ReadFile(filename)) {
    delete mesh;
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read mesh from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Faces = %d\n", mesh->NFaces());
    printf("  # Edges = %d\n", mesh->NEdges());
    printf("  # Vertices = %d\n", mesh->NVertices());
    fflush(stdout);
  }

  // Return success
  return mesh;
}

static int WriteIsosurface(R3QuadricSet *quadrics, const char *filename) {
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Write isosurface to file
  if (!quadrics->WriteIsosurfaceFile(filename, isolevel)) return 0;

  // Print statistics
  if (print_verbose) {
    printf("Wrote mesh to %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Faces = %d\n", mesh->NFaces());
    printf("  # Edges = %d\n", mesh->NEdges());
    printf("  # Vertices = %d\n", mesh->NVertices());
    fflush(stdout);
  }

  // Return success
  return 1;
}

////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////

static void LoadColor(int k, double alpha = 1) {
  // Load a unique color for each index
  // glColor3ub(255/7*(k%7), 255/11*(k%11), 255/19*(k%19));
  glColor4ub(225 / 7 * (k % 7), 225 / 11 * (k % 11), 225 / 19 * (k % 19),
             alpha * 255);
}

static void DrawText(const R3Point &p, const char *s) {
  // Draw text string s and position p
  glRasterPos3d(p[0], p[1], p[2]);
  while (*s) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *(s++));
}

static void DrawText(const R2Point &p, const char *s) {
  // Draw text string s and position p
  R3Ray ray = viewer->WorldRay((int)p[0], (int)p[1]);
  R3Point position = ray.Point(2 * viewer->Camera().Near());
  glRasterPos3d(position[0], position[1], position[2]);
  while (*s) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *(s++));
}

////////////////////////////////////////////////////////////////////////
// GLUT interface functions
////////////////////////////////////////////////////////////////////////

static void GLUTStop(void) {
  // Destroy window
  glutDestroyWindow(GLUTwindow);

  // Exit
  exit(0);
}

static void GLUTRedraw(void) {
  // Set viewing transformation
  viewer->Camera().Load();

  // Clear window
  glClearColor(background_color[0], background_color[1], background_color[2],
               1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Set lights
  static GLfloat light0_position[] = {3.0, 4.0, 5.0, 0.0};
  glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
  static GLfloat light1_position[] = {-3.0, -2.0, -3.0, 0.0};
  glLightfv(GL_LIGHT1, GL_POSITION, light1_position);

  // Compute useful stuff
  RNScalar total_weight = 0;
  RNScalar total_weighted_value = 0;
  for (int i = 0; i < quadrics->NQuadrics(); i++) {
    const R3Quadric *quadric = quadrics->Quadric(i);
    total_weight += fabs(quadric->Weight(query_position));
    total_weighted_value += fabs(quadric->WeightedValue(query_position));
  }

  // Draw isosurface faces
  if (show_isosurface_faces) {
    glEnable(GL_LIGHTING);
    glColor3d(0.8, 0.8, 0.8);
    quadrics->DrawIsoSurface(isolevel);
  }

  // Draw isosurface edges
  if (show_isosurface_edges) {
    glDisable(GL_LIGHTING);
    glColor3d(0.5, 0.0, 0.0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    quadrics->DrawIsoSurface(isolevel);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }

  // Draw quadric ellipsoids
  if (show_quadric_ellipsoids) {
    glDisable(GL_LIGHTING);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glDepthMask(FALSE);
    for (int i = 0; i < quadrics->NQuadrics(); i++) {
      const R3Quadric *quadric = quadrics->Quadric(i);
      if (i == selected_quadric_index) {
        glColor3d(0, 1, 0);
        glLineWidth(5);
      } else if (!color_by_query_support) {
        double alpha = quadric->ConstantTerm() / quadric_transparency_clamp;
        if (!show_quadric_transparency) alpha = 1.0;
        alpha = (alpha < 0) ? 0 : ((alpha > 1) ? 1 : alpha);
        LoadColor(i + 1, alpha);
      } else {
        RNScalar a =
            quadric->WeightedValue(query_position) / total_weighted_value;
        glColor3d(-4 * a, 0, 4 * a);
      }
      quadric->Outline();
      if (i == selected_quadric_index) {
        glLineWidth(1);
      }
    }
    glDepthMask(TRUE);
    glDisable(GL_BLEND);
  }

  // Draw quadric text
  if (show_quadric_text) {
    char buffer[1024];
    glDisable(GL_LIGHTING);
    for (int i = 0; i < quadrics->NQuadrics(); i++) {
      const R3Quadric *quadric = quadrics->Quadric(i);
      RNScalar a =
          quadric->WeightedValue(query_position) / total_weighted_value;
      RNScalar b = quadric->Weight(query_position) / total_weight;
      glColor3d(-4 * a, 0, 4 * a);
      // NOLINTNEXTLINE(runtime/printf)
      sprintf(buffer, "%d: %.0f %.0f", i, 1000 * a, 1000 * b);
      R3Point p = quadric->Center() + 2 * quadric->Radius(RN_Y) * R3posy_vector;
      DrawText(p, buffer);
    }
  }

  // Draw quadric samples
  if (show_quadric_samples) {
    // Create point samples (note, never deallocated)
    int nsamples = 100000;
    static R3Point *samples = NULL;
    if (!samples) {
      samples = new R3Point[nsamples];
      R3Point a = quadrics->BBox().Min();
      R3Vector b = quadrics->BBox().Max() - quadrics->BBox().Min();
      for (int i = 0; i < nsamples; i++) {
        samples[i] =
            a + R3Vector(RNRandomScalar() * b[0], RNRandomScalar() * b[1],
                         RNRandomScalar() * b[2]);
      }
    }

    // Draw/count point samples
    int npositive = 0, nnegative = 0;
    if (samples && (selected_quadric_index >= 0)) {
      const R3Quadric *quadric = quadrics->Quadric(selected_quadric_index);

      // Draw points near zero of quadric
      glDisable(GL_LIGHTING);
      glColor3d(0, 0.5, 0);
      glPointSize(3);
      glBegin(GL_POINTS);
      for (int i = 0; i < nsamples; i++) {
        RNScalar value = quadric->UnweightedValue(samples[i]);
        if (RNIsLess(value, isolevel, 1E-1))
          nnegative++;
        else if (RNIsGreater(value, isolevel, 1E-1))
          npositive++;
        else
          R3LoadPoint(samples[i]);
      }
      glEnd();
      glPointSize(1);

      // Draw text showing how many samples were positive, negative, etc.
      glColor3d(0, 0, 0);
      char buffer[1024];
      // NOLINTNEXTLINE(runtime/printf)
      sprintf(buffer, "%d   %d   %d", nnegative,
              nsamples - npositive - nnegative, npositive);
      DrawText(R2Point(50, 30), buffer);
    }
  }

  // Draw mesh faces
  if (mesh && show_mesh_faces) {
    glEnable(GL_LIGHTING);
    glColor4d(0.8, 0.8, 0.8, 0.2);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(FALSE);
    mesh->DrawFaces();
    glDepthMask(TRUE);
    glDisable(GL_BLEND);
  }

  // Draw mesh edges
  if (mesh && show_mesh_edges) {
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glDepthMask(FALSE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4d(0.0, 0.0, 0.5, 0.1);
    mesh->DrawEdges();
    glDepthMask(TRUE);
    glDisable(GL_BLEND);
  }

  // Draw bbox
  if (show_bbox) {
    glDisable(GL_LIGHTING);
    glColor3d(0.5, 0.5, 0.5);
    quadrics->BBox().Outline();
  }

  // Draw query position
  if (show_query_position) {
    glEnable(GL_LIGHTING);
    glColor3d(0, 0.5, 0);
    RNScalar r = 0.01 * quadrics->BBox().DiagonalRadius();
    R3Sphere(query_position, r).Draw();
  }

  // Draw message at bottom of screen
  if (show_message_text) {
    glDisable(GL_LIGHTING);
    glColor3d(0, 0, 0);
    char buffer[128];
    RNScalar query_value = quadrics->WeightedValue(query_position);
    // NOLINTNEXTLINE(runtime/printf)
    sprintf(buffer, "I=%f, V=%f", isolevel, query_value);
    DrawText(R2Point(50, 70), buffer);
    if (selected_quadric_index >= 0) {
      const R3Quadric *quadric = quadrics->Quadric(selected_quadric_index);
      RNScalar a = quadric->WeightedValue(query_position);
      RNScalar b = quadric->UnweightedValue(query_position);
      RNScalar c = quadric->Weight(query_position);
      RNScalar d = quadric->ConstantTerm();
      // NOLINTNEXTLINE(runtime/printf)
      sprintf(buffer, "%d  :  %f   %f   %f   %f", selected_quadric_index, a, b,
              c, d);
      DrawText(R2Point(50, 50), buffer);
    }
  }

  // Draw axes
  if (show_axes) {
    RNScalar d = quadrics->BBox().DiagonalRadius();
    glDisable(GL_LIGHTING);
    glLineWidth(3);
    R3BeginLine();
    glColor3d(1, 0, 0);
    R3LoadPoint(R3zero_point + 0.5 * d * R3negx_vector);
    R3LoadPoint(R3zero_point + d * R3posx_vector);
    R3EndLine();
    R3BeginLine();
    glColor3d(0, 1, 0);
    R3LoadPoint(R3zero_point + 0.5 * d * R3negy_vector);
    R3LoadPoint(R3zero_point + d * R3posy_vector);
    R3EndLine();
    R3BeginLine();
    glColor3d(0, 0, 1);
    R3LoadPoint(R3zero_point + 0.5 * d * R3negz_vector);
    R3LoadPoint(R3zero_point + d * R3posz_vector);
    R3EndLine();
    glLineWidth(1);
  }

  // Capture image and exit
  if (output_image_filename) {
    R2Image image(GLUTwindow_width, GLUTwindow_height, 3);
    image.Capture();
    image.Write(output_image_filename);
    GLUTStop();
  }

  // Swap buffers
  glutSwapBuffers();
}

static void GLUTResize(int w, int h) {
  // Resize window
  glViewport(0, 0, w, h);

  // Resize viewer viewport
  viewer->ResizeViewport(0, 0, w, h);

  // Remember window size
  GLUTwindow_width = w;
  GLUTwindow_height = h;

  // Redraw
  glutPostRedisplay();
}

static void GLUTMotion(int x, int y) {
  // Invert y coordinate
  y = GLUTwindow_height - y;

  // Compute mouse movement
  int dx = x - GLUTmouse[0];
  int dy = y - GLUTmouse[1];

  // Update mouse drag
  GLUTmouse_drag += dx * dx + dy * dy;

  // Check if moving query position
  if (moving_query_position) {
    // Update query position
    R3Point intersection;
    R3Ray ray = viewer->WorldRay(x, y);
    R3Vector normal = R3xyz_triad[viewer->Camera().Towards().MaxDimension()];
    R3Plane plane(query_position, normal);
    if (R3Intersects(ray, plane, &intersection)) {
      query_position = intersection;
      glutPostRedisplay();
    }
  } else {
    // World in hand navigation
    if (GLUTbutton[0])
      viewer->RotateWorld(1.0, world_origin, x, y, dx, dy);
    else if (GLUTbutton[1])
      viewer->ScaleWorld(1.0, world_origin, x, y, dx, dy);
    else if (GLUTbutton[2])
      viewer->TranslateWorld(1.0, world_origin, x, y, dx, dy);
    if (GLUTbutton[0] || GLUTbutton[1] || GLUTbutton[2]) glutPostRedisplay();
  }

  // Remember mouse position
  GLUTmouse[0] = x;
  GLUTmouse[1] = y;
}

static void GLUTMouse(int button, int state, int x, int y) {
  // Invert y coordinate
  y = GLUTwindow_height - y;

  // Process mouse button event
  // Mouse is going down
  if (state == GLUT_DOWN) {
    // Reset mouse drag
    GLUTmouse_drag = 0;

    // Process thumbwheel
    if (button == 3)
      viewer->ScaleWorld(world_origin, 0.9);
    else if (button == 4)
      viewer->ScaleWorld(world_origin, 1.1);

    // Check if selected query_position
    if (button == GLUT_LEFT_BUTTON) {
      moving_query_position = 0;
      R3Ray ray = viewer->WorldRay(x, y);
      RNScalar r = 0.01 * quadrics->BBox().DiagonalRadius();
      if (R3Intersects(ray, R3Sphere(query_position, r))) {
        moving_query_position = 1;
      }
    }
  } else if (button == GLUT_LEFT) {
    // No longer moving query position
    moving_query_position = 0;

    // Check for double click
    static RNBoolean double_click = FALSE;
    static RNTime last_mouse_down_time;
    double_click = (!double_click) && (last_mouse_down_time.Elapsed() < 0.4);
    last_mouse_down_time.Read();

    // Check for click (rather than drag)
    if (GLUTmouse_drag < 100) {
      // Select quadric
      R3Ray ray = viewer->WorldRay(x, y);
      RNLength t, closest_t = FLT_MAX;
      selected_quadric_index = -1;
      for (int i = 0; i < quadrics->NQuadrics(); i++) {
        const R3Quadric *quadric = quadrics->Quadric(i);
        if (R3Intersects(ray, quadric->Support().BBox(), NULL, NULL, &t)) {
          if (t < closest_t) {
            selected_quadric_index = i;
            closest_t = t;
          }
        }
      }
    }
  }

  // Remember button state
  int b = (button == GLUT_LEFT_BUTTON)
              ? 0
              : ((button == GLUT_MIDDLE_BUTTON) ? 1 : 2);
  GLUTbutton[b] = (state == GLUT_DOWN) ? 1 : 0;

  // Remember modifiers
  GLUTmodifiers = glutGetModifiers();

  // Remember mouse position
  GLUTmouse[0] = x;
  GLUTmouse[1] = y;

  // Redraw
  glutPostRedisplay();
}

static void GLUTSpecial(int key, int x, int y) {
  // Invert y coordinate
  y = GLUTwindow_height - y;

  // Process keyboard button event
  switch (key) {
    case GLUT_KEY_DOWN:
      if (glutGetModifiers() & GLUT_ACTIVE_SHIFT)
        isolevel -= 0.1;
      else
        isolevel -= 0.01;
      break;

    case GLUT_KEY_UP:
      if (glutGetModifiers() & GLUT_ACTIVE_SHIFT)
        isolevel += 0.1;
      else
        isolevel += 0.01;
      break;

    case GLUT_KEY_PAGE_UP:
      if (++selected_quadric_index >= quadrics->NQuadrics())
        selected_quadric_index = quadrics->NQuadrics() - 1;
      break;

    case GLUT_KEY_PAGE_DOWN:
      if (--selected_quadric_index < 0) selected_quadric_index = 0;
      break;
  }

  // Remember mouse position
  GLUTmouse[0] = x;
  GLUTmouse[1] = y;

  // Remember modifiers
  GLUTmodifiers = glutGetModifiers();

  // Redraw
  glutPostRedisplay();
}

static void GLUTKeyboard(unsigned char key, int x, int y) {
  // Process keyboard button event
  switch (key) {
    case 'A':
    case 'a':
      show_axes = !show_axes;
      break;

    case 'B':
    case 'b':
      show_bbox = !show_bbox;
      break;

    case 'C':
    case 'c':
      color_by_query_support = !color_by_query_support;
      break;

    case 'E':
    case 'e':
      show_isosurface_edges = !show_isosurface_edges;
      break;

    case 'F':
    case 'f':
      show_isosurface_faces = !show_isosurface_faces;
      break;

    case 'I':
      show_isosurface_edges = !show_isosurface_edges;
      break;

    case 'i':
      show_isosurface_faces = !show_isosurface_faces;
      break;

    case 'M':
      show_mesh_edges = !show_mesh_edges;
      break;

    case 'm':
      show_mesh_faces = !show_mesh_faces;
      break;

    case 'P':
    case 'p':
      show_query_position = !show_query_position;
      break;

    case 'Q':
    case 'q':
      show_quadric_ellipsoids = !show_quadric_ellipsoids;
      break;

    case 'S':
    case 's':
      show_quadric_samples = !show_quadric_samples;
      break;

    case 'T':
    case 't':
      show_quadric_text = !show_quadric_text;
      break;

    case 'W':
    case 'w':
      show_quadric_transparency = !show_quadric_transparency;
      break;

    case 'Z':
    case 'z':
      show_message_text = !show_message_text;
      break;

    case '~':
      quadric_transparency_clamp *= 1.1;
      break;

    case '`':
      quadric_transparency_clamp *= 0.9;
      break;

    case ' ': {
      // Print camera
      const R3Camera &camera = viewer->Camera();
      printf("#camera  %g %g %g  %g %g %g  %g %g %g  %g \n",
             camera.Origin().X(), camera.Origin().Y(), camera.Origin().Z(),
             camera.Towards().X(), camera.Towards().Y(), camera.Towards().Z(),
             camera.Up().X(), camera.Up().Y(), camera.Up().Z(), camera.YFOV());
      break;
    }

    case 27:  // ESCAPE
      GLUTStop();
      break;
  }

  // Remember mouse position
  GLUTmouse[0] = x;
  GLUTmouse[1] = GLUTwindow_height - y;

  // Remember modifiers
  GLUTmodifiers = glutGetModifiers();

  // Redraw
  glutPostRedisplay();
}

static void GLUTInit(int *argc, char **argv) {
  // Open window
  glutInit(argc, argv);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(GLUTwindow_width, GLUTwindow_height);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_ALPHA);
  GLUTwindow = glutCreateWindow("OpenGL Viewer");

  // Initialize background color
  glClearColor(200.0 / 255.0, 200.0 / 255.0, 200.0 / 255.0, 1.0);

  // Initialize lights
  static GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
  glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
  static GLfloat light0_diffuse[] = {1.0, 1.0, 1.0, 1.0};
  glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
  glEnable(GL_LIGHT0);
  static GLfloat light1_diffuse[] = {0.5, 0.5, 0.5, 1.0};
  glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse);
  glEnable(GL_LIGHT1);
  glEnable(GL_NORMALIZE);
  glEnable(GL_LIGHTING);

  // Initialize color settings
  glEnable(GL_COLOR_MATERIAL);
  glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

  // Initialize graphics modes
  glEnable(GL_DEPTH_TEST);

  // Initialize GLUT callback functions
  glutDisplayFunc(GLUTRedraw);
  glutReshapeFunc(GLUTResize);
  glutKeyboardFunc(GLUTKeyboard);
  glutSpecialFunc(GLUTSpecial);
  glutMouseFunc(GLUTMouse);
  glutMotionFunc(GLUTMotion);

  // Initialize font
#if (RN_OS == RN_WINDOWSNT)
  int font = glGenLists(256);
  wglUseFontBitmaps(wglGetCurrentDC(), 0, 256, font);
  glListBase(font);
#endif
}

static R3Viewer *CreateViewer(void) {
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Get pdb bounding box
  assert(!quadrics->BBox().IsEmpty());
  RNLength r = quadrics->BBox().DiagonalRadius();
  assert((r > 0.0) && RNIsFinite(r));

  // Setup camera view looking down the Z axis
  if (!initial_camera)
    initial_camera_origin =
        quadrics->BBox().Centroid() - initial_camera_towards * (2.5 * r);

  R3Camera camera(initial_camera_origin, initial_camera_towards,
                  initial_camera_up, 0.4, 0.4, 0.1 * r, 1000.0 * r);
  R2Viewport viewport(0, 0, GLUTwindow_width, GLUTwindow_height);
  R3Viewer *viewer = new R3Viewer(camera, viewport);

  // Print statistics
  if (print_verbose) {
    printf("Created viewer ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Origin = %g %g %g\n", camera.Origin().X(), camera.Origin().Y(),
           camera.Origin().Z());
    printf("  Towards = %g %g %g\n", camera.Towards().X(), camera.Towards().Y(),
           camera.Towards().Z());
    printf("  Up = %g %g %g\n", camera.Up().X(), camera.Up().Y(),
           camera.Up().Z());
    printf("  Fov = %g %g\n", camera.XFOV(), camera.YFOV());
    printf("  Near = %g\n", camera.Near());
    printf("  Far = %g\n", camera.Far());
    fflush(stdout);
  }

  // Return viewer
  return viewer;
}

void GLUTMainLoop(void) {
  // Set world origin
  world_origin = quadrics->BBox().Centroid();
  query_position = world_origin;

  // Create viewer
  viewer = CreateViewer();
  if (!viewer) exit(-1);

  // Run main loop -- never returns
  glutMainLoop();
}

static int ParseArgs(int argc, char **argv) {
  // Parse arguments
  argc--;
  argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) {
        print_verbose = 1;
      } else if (!strcmp(*argv, "-input_mesh")) {
        argc--;
        argv++;
        input_mesh_filename = *argv;
      } else if (!strcmp(*argv, "-output_mesh")) {
        argc--;
        argv++;
        output_isosurface_filename = *argv;
      } else if (!strcmp(*argv, "-partition_of_unity")) {
        evaluation_method = R3_QUADRIC_SET_PARTITION_OF_UNITY;
      } else if (!strcmp(*argv, "-evaluation_method")) {
        argc--;
        argv++;
        evaluation_method = atoi(*argv);  // NOLINT(runtime/deprecated_fn)
      } else if (!strcmp(*argv, "-isolevel")) {
        argc--;
        argv++;
        isolevel = atof(*argv);  // NOLINT(runtime/deprecated_fn)
      } else if (!strcmp(*argv, "-resolution")) {
        argc--;
        argv++;
        resolution = atoi(*argv);  // NOLINT(runtime/deprecated_fn)
      } else if (!strcmp(*argv, "-camera")) {
        RNCoord x, y, z, tx, ty, tz, ux, uy, uz;
        argv++;
        argc--;
        x = atof(*argv);  // NOLINT(runtime/deprecated_fn)
        argv++;
        argc--;
        y = atof(*argv);  // NOLINT(runtime/deprecated_fn)
        argv++;
        argc--;
        z = atof(*argv);  // NOLINT(runtime/deprecated_fn)
        argv++;
        argc--;
        tx = atof(*argv);  // NOLINT(runtime/deprecated_fn)
        argv++;
        argc--;
        ty = atof(*argv);  // NOLINT(runtime/deprecated_fn)
        argv++;
        argc--;
        tz = atof(*argv);  // NOLINT(runtime/deprecated_fn)
        argv++;
        argc--;
        ux = atof(*argv);  // NOLINT(runtime/deprecated_fn)
        argv++;
        argc--;
        uy = atof(*argv);  // NOLINT(runtime/deprecated_fn)
        argv++;
        argc--;
        uz = atof(*argv);  // NOLINT(runtime/deprecated_fn)
        initial_camera_origin = R3Point(x, y, z);
        initial_camera_towards.Reset(tx, ty, tz);
        initial_camera_up.Reset(ux, uy, uz);
        initial_camera = TRUE;
      } else if (!strcmp(*argv, "-window")) {
        argv++;
        argc--;
        GLUTwindow_width = atoi(*argv);  // NOLINT(runtime/deprecated_fn)
        argv++;
        argc--;
        GLUTwindow_height = atoi(*argv);  // NOLINT(runtime/deprecated_fn)
      } else if (!strcmp(*argv, "-background")) {
        argc--;
        argv++;
        background_color[0] = atof(*argv);  // NOLINT(runtime/deprecated_fn)
        argc--;
        argv++;
        background_color[1] = atof(*argv);  // NOLINT(runtime/deprecated_fn)
        argc--;
        argv++;
        background_color[2] = atof(*argv);  // NOLINT(runtime/deprecated_fn)
      } else if (!strcmp(*argv, "-image")) {
        argc--;
        argv++;
        output_image_filename = *argv;
      } else {
        RNFail("Invalid program argument: %s", *argv);
        exit(1);
      }
      argv++;
      argc--;
    } else {
      if (!input_quadrics_filename) {
        input_quadrics_filename = *argv;
      } else {
        RNFail("Invalid program argument: %s", *argv);
        exit(1);
      }
      argv++;
      argc--;
    }
  }

  // Check inputs
  if (!input_quadrics_filename) {
    RNFail("Usage: quadricview filename [options]\n");
    return 0;
  }

  // Return OK status
  return 1;
}

int main(int argc, char **argv) {
  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(-1);

  // Initialize GLUT
  GLUTInit(&argc, argv);

  // Read quadrics file
  quadrics = ReadQuadrics(input_quadrics_filename);
  if (!quadrics) exit(-1);

  // Read mesh file
  if (input_mesh_filename) {
    mesh = ReadMesh(input_mesh_filename);
    if (!mesh) exit(-1);
  }

  // Write mesh file
  if (output_isosurface_filename) {
    if (!WriteIsosurface(quadrics, output_isosurface_filename)) exit(-1);
  }

  // Run GLUT interface
  GLUTMainLoop();

  // Return success
  return 0;
}
